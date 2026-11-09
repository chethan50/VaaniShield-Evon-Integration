/* Bounded PCM capture. Wake recognition uses the explicitly selected mode.
 * Browser-default recognition may send listening audio to the browser provider.
 */
(function(root) {
  'use strict';
  const WAKE_RE = /(?:^|[\s,.:;!?।|-])((?:(?:hey|hi|hello)[ ,.!-]+(?:vaa?ni(?:shield)?|bani|wani)\b|\b(?:vaa?ni(?:shield)?)\b|(?:(?:हे|हाय|हेलो|हॅलो|ಹೇ|ಹಾಯ್|ಹಲೋ)[ ,.!।|-]+)?(?:वाणी|वानी|ವಾಣಿ)))(?=$|[\s,.:;!?।|-])/i;
  function isWake(text) { return WAKE_RE.test(String(text || '')); }
  function wakeCommand(text) {
    const raw = String(text || ''), match = WAKE_RE.exec(raw);
    return match ? raw.slice(match.index + match[0].length).replace(/^[\s,.:;!?।|-]+/, '').trim() : null;
  }
  function wav(chunks, rate) {
    const size = chunks.reduce((n, x) => n + x.length, 0);
    const samples = new Float32Array(size);
    let offset = 0;
    for (const chunk of chunks) { samples.set(chunk, offset); offset += chunk.length; }
    const count = Math.floor(size * 16000 / rate);
    const buffer = new ArrayBuffer(44 + count * 2), view = new DataView(buffer);
    const word = (at, value) => [...value].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
    word(0, 'RIFF'); view.setUint32(4, 36 + count * 2, true); word(8, 'WAVE');
    word(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, 16000, true);
    view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    word(36, 'data'); view.setUint32(40, count * 2, true);
    for (let i = 0; i < count; i++) {
      const a = Math.floor(i * rate / 16000), b = Math.max(a + 1, Math.floor((i + 1) * rate / 16000));
      let total = 0;
      for (let j = a; j < b && j < size; j++) total += samples[j];
      const s = Math.max(-1, Math.min(1, total / (b - a)));
      view.setInt16(44 + i * 2, s * (s < 0 ? 32768 : 32767), true);
    }
    return new Uint8Array(buffer);
  }
  class VoiceController {
    constructor(env) {
      this.env = env; this.state = 'IDLE'; this.enabled = false; this.generation = 0;
      this.pending = null; this.seenResponses = new Set(); this.chunks = [];
      this.installOffered = false; this.installing = false; this.installLock = env.installLock || {active:false};
      this.mode = 'on_device'; this.quickEnds = 0;
    }
    status(state, text) { this.state = state; this.statusText = text; this.env.status(state, text); }
    trace(stage, id, kind) { if (this.env.trace) this.env.trace(stage, id || null, kind || null); }
    stopRecognition() {
      clearTimeout(this.restart);
      clearTimeout(this.wakeTimer); this.wakePending = false; this.wakeAudioQueued = false;
      this.wakeDispatched = false; this.wakeEventId = null; this.wakeIndex = -1; this.wakeTranscript = '';
      if (this.recognition) {
        this.recognition.onstart = this.recognition.onend = this.recognition.onresult = this.recognition.onerror = null;
        this.recognition.abort(); this.recognition = null;
      }
    }
    release() {
      clearTimeout(this.captureTimer);
      if (this.processor) { this.processor.onaudioprocess = null; this.processor.disconnect(); this.processor = null; }
      if (this.source) { this.source.disconnect(); this.source = null; }
      if (this.stream) { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
      if (this.context) { this.context.close().catch(() => {}); this.context = null; }
    }
    interrupt() { this.generation++; this.env.stopPlayback(); this.stopRecognition(); this.release(); this.chunks = []; this.wakeChunks = []; this.wakeSamples = 0; this.finishing = false; }
    stop() {
      this.enabled = false; this.interrupt(); this.pending = null; this.pendingKind = null; this.installOffered = false;
      this.status('IDLE', 'Microphone off. Click mic to record.');
    }
    async enable(mode = 'on_device', browserConsent = false) {
      if (this.installing) return;
      this.stop(); const generation = this.generation;
      if (mode !== 'on_device' && mode !== 'browser_default') {
        this.status('UNAVAILABLE', 'Choose a valid wake recognition mode.'); return;
      }
      this.mode = mode; this.quickEnds = 0;
      if (mode === 'browser_default' && !browserConsent) {
        this.status('UNAVAILABLE', 'Consent to browser-default wake recognition in Voice Wake-Word settings before starting.'); return;
      }
      try {
        const SR = this.env.SR;
        if (!SR) { this.status('UNAVAILABLE', 'Browser wake recognition unavailable. Use the microphone button.'); return; }
        if (mode === 'browser_default') { this.enabled = true; await this.standby(); return; }
        if (!('processLocally' in SR.prototype) || typeof SR.available !== 'function') {
          this.status('UNAVAILABLE', 'On-device wake recognition unavailable. Use the microphone button.'); return;
        }
        const availability = await SR.available({langs:['en-US'], processLocally:true});
        if (generation !== this.generation) return;
        if (availability === 'downloadable' || availability === 'downloading') {
          if (typeof SR.install !== 'function') {
            this.status('UNAVAILABLE', 'The local English model is ' + availability + ', but this browser cannot install it from the app. Use push-to-talk or check browser settings.'); return;
          }
          this.installOffered = true;
          this.status('MODEL_' + availability.toUpperCase(), availability === 'downloadable'
            ? 'Local English voice model is available to download. Select Install local English voice model, or use push-to-talk.'
            : 'Local English voice model is downloading. Select Install local English voice model to wait for installation, or use push-to-talk.'); return;
        }
        if (availability !== 'available') {
          this.status('UNAVAILABLE', 'Local English voice model unavailable (' + String(availability) + '). Check browser support or use push-to-talk.'); return;
        }
        this.enabled = true; await this.standby();
      } catch (_) { if (generation === this.generation) { this.stop(); this.status('UNAVAILABLE', 'Could not check local voice model availability. Check browser settings and retry, or use push-to-talk.'); } }
    }
    async installModel() {
      if (!this.installOffered || this.installing || this.env.hidden()) return;
      if (this.installLock.active) {
        this.status('INSTALLING', 'A local voice model installation is already in progress. Wait, then retry hands-free.'); return;
      }
      const SR = this.env.SR, generation = this.generation;
      if (!SR || typeof SR.install !== 'function') {
        this.installOffered = false;
        this.status('UNAVAILABLE', 'This browser does not support installing the local voice model. Use push-to-talk.'); return;
      }
      this.installing = true; this.installLock.active = true;
      this.status('INSTALLING', 'Installing local English voice model. This may take a few minutes; push-to-talk remains available.');
      try {
        const installed = await SR.install({langs:['en-US']});
        if (generation !== this.generation || this.env.hidden()) return;
        if (installed !== true) {
          this.status('INSTALL_FAILED', 'Local voice model installation did not complete. Retry installation or use push-to-talk.'); return;
        }
        this.status('INSTALLING', 'Installation completed. Checking local model availability…');
        let availability;
        try { availability = await SR.available({langs:['en-US'], processLocally:true}); }
        catch (_) {
          if (generation === this.generation) this.status('INSTALL_FAILED', 'Could not verify the installed local voice model. Retry or use push-to-talk.');
          return;
        }
        if (generation !== this.generation || this.env.hidden()) return;
        if (availability !== 'available') {
          this.status('INSTALL_FAILED', 'Local voice model is not ready after installation (' + String(availability) + '). Retry or use push-to-talk.'); return;
        }
        this.installOffered = false;
        this.status('MODEL_READY', 'Local English voice model installed. Starting hands-free listening…');
        this.enabled = true; await this.standby();
        if (generation === this.generation && this.state === 'STANDBY')
          this.status('STANDBY', 'Local English voice model installed · Mic active · Say Hey Vaani (on-device)');
      } catch (_) {
        if (generation === this.generation) this.status('INSTALL_FAILED', 'Local voice model installation failed or was blocked. Retry or use push-to-talk.');
      } finally {
        this.installing = false; this.installLock.active = false;
        this.env.status(this.state, this.statusText);
      }
    }
    async standby(duringPlayback = false) {
      if (!this.enabled || this.env.hidden()) return;
      const generation = this.generation;
      if (this.standbyTask && this.standbyTask.generation === generation) {
        await this.standbyTask.promise;
        if (!duringPlayback && generation === this.generation && this.enabled && this.recognition)
          this.status('STANDBY', 'Mic active · Say Hey Vaani (' + (this.mode === 'on_device' ? 'on-device' : 'browser-default') + ')');
        return;
      }
      const task = {generation, promise:this.startStandby(duringPlayback)};
      this.standbyTask = task;
      try { await task.promise; }
      finally { if (this.standbyTask === task) this.standbyTask = null; }
    }
    async startStandby(duringPlayback = false) {
      if (!this.enabled || this.env.hidden()) return;
      this.stopRecognition();
      const generation = this.generation;
      if (!this.stream) {
        this.status('STARTING', 'Requesting microphone permission for wake listening…');
        try {
          const stream = await this.env.getUserMedia({audio:{echoCancellation:true, noiseSuppression:true, channelCount:1}});
          if (generation !== this.generation || !this.enabled || this.env.hidden()) { stream.getTracks().forEach(t => t.stop()); return; }
          this.stream = stream; this.context = new this.env.AudioContext(); await this.context.resume();
          if (generation !== this.generation || !this.enabled || this.env.hidden()) return;
          this.rate = this.context.sampleRate; this.wakeChunks = []; this.wakeSamples = 0;
          this.source = this.context.createMediaStreamSource(stream);
          this.processor = this.context.createScriptProcessor(4096, 1, 1);
          this.processor.onaudioprocess = event => this.captureWake(event.inputBuffer.getChannelData(0));
          this.source.connect(this.processor); this.processor.connect(this.context.destination);
        } catch (error) {
          if (generation !== this.generation) return;
          this.stop(); this.status('ERROR', error && (error.name === 'NotAllowedError' || error.name === 'SecurityError')
            ? 'Microphone permission denied. Allow access and start hands-free again.'
            : 'Cannot start the microphone. Check the device and permission, then retry.'); return;
        }
      }
      if (generation !== this.generation || !this.enabled || this.env.hidden()) return;
      this.stopRecognition();
      let rec;
      try { rec = new this.env.SR(); }
      catch (_) { this.stop(); this.status('UNAVAILABLE', 'Wake recognition failed to initialize. Retry or use the microphone button.'); return; }
      if (this.mode === 'on_device') rec.processLocally = true;
      rec.lang = 'en-IN'; rec.continuous = true; rec.interimResults = true;
      let sessionStarted = Date.now(), hadNoSpeech = false, hadActivity = false;
      rec.onstart = () => { sessionStarted = Date.now(); hadNoSpeech = false; };
      rec.onresult = event => {
        if (!this.enabled || generation !== this.generation || this.recognition !== rec) return;
        if (this.wakeDispatched || this.state === 'RECORDING') {
          this.trace('duplicate_event_rejected', this.wakeEventId || this.recordingId, 'wake');
          return;
        }
        hadActivity = true;
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          if (!result || !result[0]) continue;
          const transcript = result[0].transcript || '', command = wakeCommand(transcript);
          if (command !== null) {
            if (!this.wakePending) {
              this.wakePending = true; this.wakeIndex = i; this.wakeTranscript = transcript;
              this.wakeEventId = this.env.id();
              this.trace('wake_phrase_detected', this.wakeEventId, 'wake');
              this.status('WAKE_DETECTED', 'Wake phrase detected · Listening for your command…');
              if (command) { this.queueWakeAudio(generation); return; }
              clearTimeout(this.wakeTimer);
              this.wakeTimer = setTimeout(() => {
                if (generation !== this.generation || !this.wakePending || this.wakeDispatched) return;
                this.wakeDispatched = true;
                const wakeId = this.wakeEventId;
                this.interrupt();
                if (duringPlayback) this.record(true, wakeId);
                else { this.status('PROCESSING', 'Wake phrase detected · Preparing response…'); this.send({kind:'wake'}, wakeId); }
              }, 550);
            } else {
              this.wakeTranscript = transcript;
              if (command) { this.queueWakeAudio(generation); return; }
              this.trace('duplicate_event_rejected', this.wakeEventId, 'wake');
            }
          } else if (this.wakePending && i > this.wakeIndex && transcript.trim()) {
            this.queueWakeAudio(generation); return;
          }
        }
      };
      rec.onerror = event => {
        this.trace('recognition_error', null, event.error);
        if (event.error === 'no-speech') { hadNoSpeech = true; return; }
        if (event.error === 'aborted') return;
        this.stop();
        const message = event.error === 'not-allowed' || event.error === 'service-not-allowed'
          ? 'Microphone or speech recognition permission denied. Allow it in browser settings, then retry.'
          : event.error === 'audio-capture' ? 'No microphone available. Check your input device, then retry.'
          : 'Wake recognition stopped (' + event.error + '). Retry hands-free or use push-to-talk.';
        this.status('UNAVAILABLE', message);
      };
      rec.onend = () => {
        this.trace('recognition_end');
        if (this.enabled && generation === this.generation && this.recognition === rec) {
          // Some recognizers end immediately after delivering their final result.
          // Let the pending wake/command finish before treating this as a stop.
          if (this.wakePending || this.wakeAudioQueued || this.wakeDispatched) { this.recognition = null; return; }
          this.recognition = null;
          const elapsed = Date.now() - sessionStarted;
          this.quickEnds = (!hadNoSpeech && !hadActivity && elapsed < 300) ? this.quickEnds + 1 : 0;
          if (this.quickEnds >= 3) {
            this.stop(); this.status('ERROR', 'Wake recognition stopped repeatedly. Check browser permissions and start hands-free again.'); return;
          }
          if (!duringPlayback) this.status('STARTING', 'Wake recognition paused. Restarting listening…');
          this.restart = setTimeout(() => {
            if (this.enabled && generation === this.generation && !this.env.hidden()) this.standby(duringPlayback);
          }, 500);
        }
      };
      this.recognition = rec;
      try {
        rec.start();
        this.trace('recognition_started');
        if (!duringPlayback) this.status('STANDBY', 'Mic active · Say Hey Vaani (' + (this.mode === 'on_device' ? 'on-device' : 'browser-default') + ')');
      } catch (_) { this.stop(); this.status('UNAVAILABLE', 'Wake recognition failed to initialize. Retry or use push-to-talk.'); }
    }
    captureWake(data) {
      if (!this.enabled || this.env.hidden()) return;
      this.wakeChunks.push(new Float32Array(data)); this.wakeSamples += data.length;
      // Keep only a short rolling buffer in memory; it is discarded on stop.
      while (this.wakeChunks.length > 1 && this.wakeSamples - this.wakeChunks[0].length >= this.rate * 8)
        this.wakeSamples -= this.wakeChunks.shift().length;
    }
    queueWakeAudio(generation) {
      if (this.wakeDispatched || this.state === 'RECORDING') {
        this.trace('duplicate_event_rejected', this.wakeEventId || this.recordingId, 'audio');
        return;
      }
      if (this.wakeAudioQueued) {
        this.trace('duplicate_event_rejected', this.wakeEventId, 'audio');
        return;
      }
      this.wakeAudioQueued = true; clearTimeout(this.wakeTimer);
      const eventId = this.wakeEventId || (this.wakeEventId = this.env.id());
      this.trace('wake_command_recording_started', eventId, 'wake_command');
      this.status('CAPTURING', 'Wake phrase detected · Capturing command…');
      this.wakeTimer = setTimeout(() => {
        this.wakeAudioQueued = false;
        if (generation !== this.generation || !this.enabled || this.env.hidden() || this.wakeDispatched) return;
        this.wakeDispatched = true;
        const chunks = this.wakeChunks || [], rate = this.rate;
        this.interrupt();
        if (!chunks.length) { this.status('ERROR', 'No command audio captured. Click the microphone and try again.'); return; }
        const bytes = wav(chunks, rate);
        this.trace('wake_command_recording_stopped', eventId, 'wake_command');
        this.trace('audio_event_created', eventId, 'audio');
        this.status('PROCESSING', 'Uploading audio · Transcribing and preparing your response…');
        this.send({kind:'audio', audio:this.env.base64(bytes)}, eventId);
      }, 250);
    }
    async record(automatic = false, eventId = null) {
      this.interrupt(); this.pending = null;
      this.recordingId = eventId || this.env.id();
      const startStage = automatic ? 'wake_command_recording_started' : 'push_to_talk_recording_started';
      const kindLabel = automatic ? 'wake_followup' : 'push_to_talk';
      this.trace(startStage, this.recordingId, kindLabel);
      const generation = this.generation;
      this.status('STARTING', 'Requesting microphone permission…');
      try {
        const stream = await this.env.getUserMedia({audio:{echoCancellation:true, noiseSuppression:true, channelCount:1}});
        if (generation !== this.generation || this.env.hidden()) { stream.getTracks().forEach(t => t.stop()); return; }
        this.stream = stream;
        this.context = new this.env.AudioContext(); await this.context.resume();
        if (generation !== this.generation) return;
        this.rate = this.context.sampleRate; this.chunks = []; this.samples = 0;
        this.speechMs = 0; this.silenceMs = 0; this.automatic = automatic;
        this.source = this.context.createMediaStreamSource(stream);
        // ScriptProcessor is widely available in the supported Streamlit browsers.
        // It processes locally and is disconnected immediately after a bounded turn.
        this.processor = this.context.createScriptProcessor(4096, 1, 1);
        this.processor.onaudioprocess = event => this.capture(event.inputBuffer.getChannelData(0));
        this.source.connect(this.processor); this.processor.connect(this.context.destination);
        this.captureTimer = setTimeout(() => this.finish(), 30000);
        this.status('RECORDING', 'Mic active · Recording · Click mic to send, Cancel to discard');
      } catch (_) { this.stop(); this.status('ERROR', 'Cannot access microphone. Check permission/device and retry.'); }
    }
    capture(data) {
      if (this.state !== 'RECORDING') return;
      this.chunks.push(new Float32Array(data)); this.samples += data.length;
      const ms = data.length / this.rate * 1000;
      const rms = Math.sqrt(data.reduce((s, x) => s + x * x, 0) / data.length);
      if (rms > 0.018) { this.speechMs += ms; this.silenceMs = 0; } else this.silenceMs += ms;
      if (this.samples / this.rate >= 30 ||
          (this.automatic && this.speechMs >= 200 && this.silenceMs >= 1100)) this.finish();
      else if (this.automatic && this.speechMs === 0 && this.samples / this.rate >= 10) {
        this.interrupt(); this.standby();
      }
    }
    finish() {
      if (this.state !== 'RECORDING' || this.finishing) {
        if (this.recordingId) this.trace('duplicate_event_rejected', this.recordingId, this.automatic ? 'wake_followup' : 'push_to_talk');
        return;
      }
      this.finishing = true;
      const chunks = this.chunks; this.chunks = []; this.release();
      if (!chunks.length || this.speechMs < 100) {
        this.finishing = false;
        this.status('ERROR', 'No speech captured. Click mic to retry.'); return;
      }
      const bytes = wav(chunks, this.rate);
      const stopStage = this.automatic ? 'wake_command_recording_stopped' : 'push_to_talk_recording_stopped';
      const kindLabel = this.automatic ? 'wake_followup' : 'push_to_talk';
      this.trace(stopStage, this.recordingId, kindLabel);
      this.trace('audio_event_created', this.recordingId, 'audio');
      this.status('PROCESSING', 'Uploading audio · Transcribing and preparing your response…');
      this.send({kind:'audio', audio:this.env.base64(bytes)}, this.recordingId);
    }
    send(value, recordingId) {
      const id = recordingId || this.env.id(); this.pending = id; this.pendingKind = value.kind;
      this.trace('component_event_sent', id, value.kind);
      this.env.emit({...value, id});
    }
    async response(result) {
      if (!result) return;
      if (this.seenResponses.has(result.id)) {
        this.trace('duplicate_event_rejected', result.id, 'response');
        return;
      }
      this.seenResponses.add(result.id);
      if (result.id !== this.pending) return; // stopped/interrupted response must never speak
      const pendingKind = this.pendingKind;
      this.pending = null; this.pendingKind = null; const generation = this.generation;
      this.trace('response_received', result.id, pendingKind);
      if (result.error) { this.status('ERROR', result.message || 'Voice request failed. Use Retry this message or record again.'); return; }
      if (result.audio) {
        this.env.stopPlayback();
        this.status('SPEAKING', 'Speaking · Say Hey Vaani to interrupt, or click mic / Stop');
        if (this.enabled) this.standby(true);
        this.trace('response_playback_started', result.id, pendingKind);
        try { await this.env.play(result.audio, result.media_type); }
        catch (_) {
          if (generation === this.generation) { this.stopRecognition(); this.status('ERROR', 'Playback blocked. Use the response audio controls.'); }
          return;
        }
        if (generation === this.generation) this.trace('response_playback_completed', result.id, pendingKind);
      }
      if (generation !== this.generation) return;
      this.stopRecognition();
      // Only a wake-only greeting requests a follow-up command. Normal voice
      // answers return to wake standby instead of recording unsolicited audio.
      if (this.enabled && pendingKind === 'wake') this.record(true);
      else if (this.enabled) this.standby();
      else this.status('IDLE', 'Microphone off. Click mic to record.');
    }
  }
  if (typeof module !== 'undefined') { module.exports = {VoiceController, wav, isWake, wakeCommand}; return; }
  const parent = window.parent, doc = parent.document, mic = document.getElementById('mic');
  let audio = null, resolvePlayback = null, allowed = false, scope = null, mode = 'on_device', browserConsent = false, disposed = false;
  const post = (type, extra) => parent.postMessage({isStreamlitMessage:true,type,...extra}, '*');
  const toolbar = doc.createElement('div'); toolbar.className = 'vaani-compact-voice';
  const label = doc.createElement('span'); label.setAttribute('role','status'); label.setAttribute('aria-live','polite');
  const modeLabel = doc.createElement('span'); modeLabel.textContent = 'Recognition: strict on-device';
  const listen = doc.createElement('button'); listen.textContent = 'Start hands-free';
  listen.title = 'Start wake listening in the selected mode while this tab is active.';
  const install = doc.createElement('button'); install.textContent = 'Install local English voice model';
  install.title = 'Download the browser’s on-device English voice model.';
  const stop = doc.createElement('button'); stop.textContent = 'Stop / mute';
  const cancel = doc.createElement('button'); cancel.textContent = 'Cancel recording';
  toolbar.append(modeLabel, label, listen, install, stop, cancel);
  let msgCount = 0, userScrolledUp = false;
  function setupCopilotScrollAndComposer() {
    const area = doc.querySelector('.st-key-copilot-chat-scroll-area');
    if (!area) return;
    if (!area.dataset.vaaniScrollBound) {
      area.dataset.vaaniScrollBound = 'true';
      area.addEventListener('scroll', () => { userScrolledUp = area.scrollHeight - area.scrollTop - area.clientHeight > 65; }, {passive:true});
    }
    const count = area.querySelectorAll('[data-testid="stChatMessage"]').length;
    if (count > msgCount || !userScrolledUp) area.scrollTop = area.scrollHeight;
    msgCount = count;
  }
  const frame = window.frameElement;
  const scrollObserver = new MutationObserver(() => {
    if (frame && !frame.isConnected) cleanup();
    else setupCopilotScrollAndComposer();
  });
  scrollObserver.observe(doc.body,{childList:true,subtree:true});
  const controller = new VoiceController({
    SR:parent.SpeechRecognition || parent.webkitSpeechRecognition,
    AudioContext:parent.AudioContext || parent.webkitAudioContext,
    installLock:parent.__vaaniVoiceInstallLock || (parent.__vaaniVoiceInstallLock = {active:false}),
    hidden:() => doc.hidden,
    getUserMedia: constraints => navigator.mediaDevices.getUserMedia(constraints),
    id:() => crypto.randomUUID(),
    base64: bytes => { let s = ''; for (let i=0;i<bytes.length;i+=8192) s += String.fromCharCode(...bytes.subarray(i,i+8192)); return btoa(s); },
    trace:(stage, id, kind) => {
      const log = parent.__vaaniVoiceDiagnostics || (parent.__vaaniVoiceDiagnostics = []);
      log.push({stage, event_id:id || null, kind:kind || null});
      if (log.length > 100) log.splice(0, log.length - 100);
      if (parent.console && typeof parent.console.info === 'function') parent.console.info('[Vaani voice]', {stage, event_id:id, kind});
    },
    emit:value => post('streamlit:setComponentValue', {value, dataType:'json'}),
    status:(state, text) => {
      label.textContent = text; mic.dataset.recording = String(state === 'RECORDING');
      mic.textContent = state === 'RECORDING' ? '■' : '🎙';
      mic.setAttribute('aria-label', state === 'RECORDING' ? 'Stop recording and send' : 'Record voice message');
      cancel.hidden = state !== 'RECORDING' && state !== 'CAPTURING'; stop.hidden = state === 'IDLE';
      listen.hidden = !allowed || controller.enabled; listen.disabled = controller.installing;
      install.hidden = !allowed || !controller.installOffered; install.disabled = controller.installing;
    },
    stopPlayback:() => {
      if (audio) { audio.pause(); audio.src = ''; audio = null; }
      if (resolvePlayback) { resolvePlayback(); resolvePlayback = null; }
      doc.querySelectorAll('audio').forEach(el => { el.pause(); el.removeAttribute('autoplay'); });
    },
    play:(encoded, media) => new Promise((resolve, reject) => {
      audio = new Audio('data:' + (media || 'audio/wav') + ';base64,' + encoded);
      resolvePlayback = resolve; audio.onended = resolve; audio.onerror = reject;
      audio.play().catch(reject);
    }),
  });
  let resumePrevious = null;
  if (parent.__vaaniVoiceManager) {
    const previous = parent.__vaaniVoiceManager;
    if (previous.enabled && ['STANDBY', 'STARTING'].includes(previous.state) && !doc.hidden)
      resumePrevious = {mode:previous.mode, scope:previous.voiceScope};
    if (typeof previous.dispose === 'function') previous.dispose();
    else if (typeof previous.stop === 'function') previous.stop();
    else if (typeof previous.stopAll === 'function') previous.stopAll();
  }
  if (parent.__vaaniScrollInterval) { parent.clearInterval(parent.__vaaniScrollInterval); parent.__vaaniScrollInterval = null; }
  parent.__vaaniVoiceManager = controller;
  mic.onclick = () => controller.state === 'RECORDING' ? controller.finish() : controller.record();
  listen.onclick = () => { if (allowed) controller.enable(mode, browserConsent); };
  install.onclick = () => { if (allowed) controller.installModel(); };
  stop.onclick = () => controller.stop();
  cancel.onclick = () => { controller.interrupt(); controller.status('IDLE', 'Recording discarded. Click mic to retry.'); if(controller.enabled) controller.standby(); };
  const onHide = () => { if (doc.hidden) { controller.stop(); controller.status('IDLE', 'Hands-free stopped while the tab is hidden. Return and select Start hands-free to reactivate.'); } };
  const onText = event => {
    if (event.target.closest('.st-key-copilot_send_button') ||
        (event.key === 'Enter' && !event.shiftKey && event.target.closest('.st-key-copilot-composer-pill textarea'))) {
      controller.interrupt(); controller.pending = null; controller.pendingKind = null;
      if (controller.enabled && !doc.hidden) {
        controller.status('STARTING', 'Text sent · Returning to wake listening…'); controller.standby();
      } else controller.status('IDLE', 'Microphone off. Click mic to record.');
    }
  };
  doc.addEventListener('visibilitychange', onHide);
  doc.addEventListener('click', onText, true); doc.addEventListener('keydown', onText, true);
  function cleanup() {
    if (disposed) return;
    disposed = true;
    controller.stop(); scrollObserver.disconnect(); toolbar.remove();
    doc.removeEventListener('visibilitychange',onHide);
    doc.removeEventListener('click',onText,true); doc.removeEventListener('keydown',onText,true);
    if (parent.__vaaniVoiceManager === controller) delete parent.__vaaniVoiceManager;
  }
  controller.dispose = cleanup;
  window.addEventListener('pagehide', cleanup);
  window.addEventListener('unload', cleanup);
  window.addEventListener('message', event => {
    if (event.source !== parent || event.data.type !== 'streamlit:render') return;
    const args = event.data.args;
    if (scope && scope !== args.scope) controller.stop();
    scope = args.scope;
    controller.voiceScope = scope;
    const nextMode = args.recognition_mode === 'browser_default' ? 'browser_default' : 'on_device';
    const nextConsent = !!args.browser_consent;
    if (mode !== nextMode || (mode === 'browser_default' && browserConsent && !nextConsent)) controller.stop();
    mode = nextMode; browserConsent = nextConsent;
    modeLabel.textContent = mode === 'on_device' ? 'Recognition: strict on-device' : 'Recognition: browser-default (audio may be sent to browser provider)';
    if (allowed && !args.hands_free_allowed) controller.stop();
    allowed = !!args.hands_free_allowed; listen.hidden = !allowed || controller.enabled;
    install.hidden = !allowed || !controller.installOffered;
    if (resumePrevious) {
      const prior = resumePrevious; resumePrevious = null;
      if (allowed && !doc.hidden && prior.scope === scope && prior.mode === mode &&
          (mode !== 'browser_default' || browserConsent)) controller.enable(mode, browserConsent);
    }
    const host = doc.querySelector('.st-key-copilot-sticky-composer');
    if (host && !toolbar.isConnected) host.append(toolbar);
    controller.response(args.response);
    post('streamlit:setFrameHeight', {height:42});
  });
  controller.status('IDLE','Microphone off. Click mic to record.');
  post('streamlit:componentReady',{apiVersion:1});
})(globalThis);
