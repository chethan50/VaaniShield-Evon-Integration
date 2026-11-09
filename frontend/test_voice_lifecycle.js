'use strict';
const assert = require('node:assert/strict');
const {test} = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const {VoiceController, wav, isWake, wakeCommand} = require('./components/voice_capture/voice.js');

function harness(options = {}) {
  const sent = [], states = [], traces = []; let stops = 0, plays = 0, nextId = 0, releasePlayback;
  class SR {
    static available = async () => 'available';
    start() { this.started = true; }
    abort() { this.aborted = true; }
  }
  SR.prototype.processLocally = false;
  class AudioContext {
    constructor() { this.sampleRate = 16000; this.destination = {}; }
    async resume() {} async close() {}
    createMediaStreamSource() { return {connect(){}, disconnect(){}}; }
    createScriptProcessor() { return {connect(){}, disconnect(){}}; }
  }
  const env = {
    SR, AudioContext, hidden:() => false, status:(...s) => states.push(s),
    trace:(stage, id, kind) => traces.push({stage, event_id:id, kind}),
    id:() => String(++nextId), emit:x => sent.push(x), base64:x => Buffer.from(x).toString('base64'),
    stopPlayback:() => { stops++; if (releasePlayback) releasePlayback(); },
    play:() => { plays++; return new Promise(r => { releasePlayback = r; }); },
    getUserMedia:async () => ({getTracks:() => [{stop(){}}]}), ...options,
  };
  return {c:new VoiceController(env), env, sent, states, traces, stops:() => stops, plays:() => plays, finishPlayback:() => releasePlayback()};
}
function modelHarness(available, install) {
  class SR {
    static available = available;
    start() { this.started = true; }
    abort() { this.aborted = true; }
  }
  SR.prototype.processLocally = false;
  if (install !== undefined) SR.install = install;
  return harness({SR});
}
test('browser-default requires consent and works without local recognition APIs', async () => {
  class SR { start() { this.started = true; } abort() { this.aborted = true; } }
  const h = harness({SR}); await h.c.enable('browser_default', false);
  assert.equal(h.c.enabled, false); assert.equal(h.c.recognition, undefined); assert.match(h.states.at(-1)[1], /consent/i);
  await h.c.enable('browser_default', true);
  assert.equal(h.c.enabled, true); assert.equal(h.c.recognition.started, true);
  assert.equal('processLocally' in h.c.recognition, false); assert.equal(h.c.recognition.lang, 'en-IN');
  assert.match(h.states.at(-1)[1], /browser-default/); h.c.stop();
});
test('browser-default wake and command submit audio once through existing bridge', async () => {
  class SR { start() {} abort() {} }
  const h = harness({SR}); await h.c.enable('browser_default', true);
  h.c.captureWake(new Float32Array(4000).fill(0.1));
  const rec = h.c.recognition, result = {resultIndex:0, results:[[{transcript:'Hey Vaani, show my balance'}]]};
  rec.onresult(result); rec.onresult(result); await new Promise(r => setTimeout(r, 300));
  assert.equal(h.sent.length, 1); assert.equal(h.sent[0].kind, 'audio');
  assert.equal('text' in h.sent[0], false); h.c.stop();
});
test('local failure never falls back to browser-default recognition', async () => {
  let starts = 0; class SR { static async available() { return 'unavailable'; } start() { starts++; } abort() {} }
  SR.prototype.processLocally = false;
  const h = harness({SR}); await h.c.enable('on_device', true);
  assert.equal(h.c.enabled, false); assert.equal(starts, 0); assert.equal(h.c.mode, 'on_device');
});
test('normal recognition end restarts once and repeated quick endings stop', async () => {
  const h = harness(); await h.c.enable(); const first = h.c.recognition;
  first.onend(); await new Promise(r => setTimeout(r, 540));
  assert.ok(h.c.recognition); assert.notEqual(h.c.recognition, first);
  h.c.recognition.onend(); await new Promise(r => setTimeout(r, 540));
  h.c.recognition.onend(); assert.equal(h.c.enabled, false);
  assert.match(h.states.at(-1)[1], /stopped repeatedly/i); h.c.stop();
});
test('permission and device errors stop without restarting', async () => {
  for (const error of ['not-allowed', 'service-not-allowed', 'audio-capture']) {
    const h = harness(); await h.c.enable(); const rec = h.c.recognition;
    rec.onerror({error}); assert.equal(h.c.enabled, false); assert.equal(rec.onend, null);
    assert.match(h.states.at(-1)[1], error === 'audio-capture' ? /microphone available/i : /permission denied/i); h.c.stop();
  }
});
test('available local model starts only after hands-free activation', async () => {
  let calls = 0; const h = modelHarness(async options => {
    calls++; assert.deepEqual(options, {langs:['en-US'], processLocally:true}); return 'available';
  }, async () => { throw new Error('must not install'); });
  assert.equal(calls, 0); await h.c.enable();
  assert.equal(calls, 1); assert.equal(h.c.enabled, true); assert.equal(h.c.recognition.processLocally, true); h.c.stop();
});
for (const availability of ['downloadable', 'downloading']) {
  test(availability + ' offers explicit installation without automatically downloading', async () => {
    let installs = 0; const h = modelHarness(async () => availability, async () => { installs++; return true; });
    await h.c.enable(); assert.equal(installs, 0); assert.equal(h.c.installOffered, true);
    assert.equal(h.c.state, 'MODEL_' + availability.toUpperCase()); assert.equal(h.c.enabled, false);
    assert.equal(h.c.recognition, undefined); h.c.stop();
  });
}
test('unavailable model does not offer installation or start recognition', async () => {
  const h = modelHarness(async () => 'unavailable', async () => true); await h.c.enable();
  assert.equal(h.c.installOffered, false); assert.equal(h.c.enabled, false); assert.match(h.states.at(-1)[1], /unavailable/);
});
test('installation success rechecks local availability and enables one listener', async () => {
  let checks = 0, installs = 0;
  const h = modelHarness(async options => {
    assert.deepEqual(options, {langs:['en-US'], processLocally:true}); return ++checks === 1 ? 'downloadable' : 'available';
  }, async options => { installs++; assert.deepEqual(options, {langs:['en-US']}); return true; });
  await h.c.enable(); await h.c.installModel();
  assert.equal(installs, 1); assert.equal(checks, 2); assert.equal(h.c.enabled, true);
  assert.equal(h.c.recognition.processLocally, true); assert.equal(h.c.state, 'STANDBY');
  assert.match(h.states.at(-1)[1], /installed/i); h.c.stop();
});
for (const outcome of ['false', 'unavailable', 'reject', 'not-ready', 'check-reject']) {
  test('installation ' + outcome + ' reports failure without enabling recognition', async () => {
    let checks = 0;
    const h = modelHarness(async () => {
      if (++checks === 1) return 'downloadable';
      if (outcome === 'check-reject') throw new Error('check failed');
      return outcome === 'not-ready' ? 'downloading' : 'available';
    }, async () => { if (outcome === 'reject') throw new Error('failed'); return outcome === 'unavailable' ? 'unavailable' : outcome !== 'false'; });
    await h.c.enable(); await h.c.installModel();
    assert.equal(h.c.state, 'INSTALL_FAILED'); assert.equal(h.c.enabled, false); assert.equal(h.c.recognition, undefined);
    assert.equal(h.c.installOffered, true); assert.equal(h.c.installing, false);
  });
}
test('missing install API explains limitation and keeps push-to-talk usable', async () => {
  const h = modelHarness(async () => 'downloadable'); await h.c.enable();
  assert.equal(h.c.installOffered, false); assert.match(h.states.at(-1)[1], /cannot install/i);
  await h.c.record(); assert.equal(h.c.state, 'RECORDING');
  h.c.capture(new Float32Array(4000).fill(0.1)); h.c.finish();
  assert.equal(h.sent.length, 1); assert.equal(h.sent[0].kind, 'audio'); h.c.stop();
});
test('failed availability check explains failure without starting recognition', async () => {
  const h = modelHarness(async () => { throw new Error('blocked'); }, async () => true);
  await h.c.enable(); assert.equal(h.c.enabled, false); assert.equal(h.c.installOffered, false);
  assert.match(h.states.at(-1)[1], /could not check/i);
});
test('repeated install and start clicks cannot launch concurrent installation or listeners', async () => {
  let resolve, installs = 0, checks = 0;
  const h = modelHarness(async () => ++checks === 1 ? 'downloadable' : 'available',
    () => { installs++; return new Promise(r => { resolve = r; }); });
  await h.c.enable(); const first = h.c.installModel();
  await h.c.installModel(); await h.c.enable(); assert.equal(installs, 1); assert.equal(checks, 1);
  resolve(true); await first; assert.equal(h.c.enabled, true); assert.equal(checks, 2); h.c.stop();
});
test('stopping during installation prevents its completion from starting a listener', async () => {
  let resolve; const h = modelHarness(async () => 'downloadable', () => new Promise(r => { resolve = r; }));
  await h.c.enable(); const pending = h.c.installModel(); h.c.stop(); resolve(true); await pending;
  assert.equal(h.c.enabled, false); assert.equal(h.c.recognition, undefined); assert.equal(h.c.state, 'IDLE');
});
test('a replacement controller cannot start a second installation while the first is pending', async () => {
  let resolve, calls = 0; const lock = {active:false};
  class SR { static async available() { return 'downloadable'; }
    static install() { calls++; return new Promise(r => { resolve = r; }); } }
  SR.prototype.processLocally = false;
  const first = harness({SR, installLock:lock}), second = harness({SR, installLock:lock});
  await first.c.enable(); const pending = first.c.installModel(); first.c.stop();
  await second.c.enable(); await second.c.installModel(); assert.equal(calls, 1);
  assert.match(second.states.at(-1)[1], /already in progress/i);
  resolve(false); await pending; assert.equal(lock.active, false); second.c.stop();
});
test('component exposes installation only after a hands-free request and explicit click', async () => {
  let installs = 0, finishInstall, checks = 0;
  class SR {
    static async available() { return ++checks === 1 ? 'downloadable' : 'available'; }
    static install() { installs++; return new Promise(resolve => { finishInstall = resolve; }); }
    start() {} abort() {}
  }
  SR.prototype.processLocally = false;
  const elements = [], handlers = {}, mic = {dataset:{}, setAttribute(){}};
  const makeElement = tag => ({tag, children:[], dataset:{}, setAttribute(){}, append(...items){this.children.push(...items);}, remove(){}});
  const doc = {hidden:false, body:{}, createElement(tag){const el = makeElement(tag); elements.push(el); return el;},
    querySelector(){return null;}, querySelectorAll(){return [];}, addEventListener(){}, removeEventListener(){}};
  const parent = {document:doc, webkitSpeechRecognition:SR, AudioContext:harness().env.AudioContext, postMessage(){}, clearInterval(){}};
  const window = {parent, frameElement:{isConnected:true}, addEventListener(name, fn){handlers[name] = fn;}};
  const sandbox = {window, document:{getElementById(){return mic;}}, navigator:{mediaDevices:{getUserMedia:async () => ({getTracks:() => [{stop(){}}]})}},
    MutationObserver:class {observe(){} disconnect(){}}, crypto:{randomUUID:() => 'id'}, setTimeout, clearTimeout, Audio:class {}};
  vm.runInNewContext(fs.readFileSync(require.resolve('./components/voice_capture/voice.js'), 'utf8'), sandbox);
  const listen = elements.find(el => el.textContent === 'Start hands-free');
  const install = elements.find(el => el.textContent === 'Install local English voice model');
  assert.equal(install.hidden, true); assert.equal(installs, 0);
  handlers.message({source:parent, data:{type:'streamlit:render', args:{scope:'one', hands_free_allowed:true}}});
  listen.onclick(); await new Promise(r => setImmediate(r));
  assert.equal(install.hidden, false); assert.equal(installs, 0);
  install.onclick(); install.onclick(); assert.equal(installs, 1); assert.equal(install.disabled, true);
  finishInstall(false); await new Promise(r => setImmediate(r));
  assert.equal(install.disabled, false); assert.equal(parent.__vaaniVoiceManager.enabled, false);
  parent.__vaaniVoiceManager.stop();
});
test('component enforces consent, text stop, hidden-tab stop and replacement cleanup', async () => {
  let starts = 0, aborts = 0; const listeners = {};
  class SR { start() { starts++; } abort() { aborts++; } }
  const doc = {hidden:false, body:{}, createElement(){return {dataset:{}, setAttribute(){}, append(){}, remove(){}};},
    querySelector(){return null;}, querySelectorAll(){return [];},
    addEventListener(name, fn){(listeners[name] ||= new Set()).add(fn);},
    removeEventListener(name, fn){listeners[name].delete(fn);}};
  const parent = {document:doc, SpeechRecognition:SR, AudioContext:harness().env.AudioContext, postMessage(){}, clearInterval(){}};
  function mount() {
    const handlers = {}, elements = [], mic = {dataset:{},setAttribute(){}};
    const create = doc.createElement; doc.createElement = function() { const el = create(); elements.push(el); return el; };
    const window = {parent, frameElement:{isConnected:true}, addEventListener(name, fn){handlers[name] = fn;}};
    const sandbox = {window, document:{getElementById(){return mic;}}, navigator:{mediaDevices:{getUserMedia:async () => ({getTracks:() => [{stop(){}}]})}},
      MutationObserver:class {observe(){} disconnect(){}}, crypto:{randomUUID:() => 'id'}, setTimeout, clearTimeout, Audio:class {}};
    vm.runInNewContext(fs.readFileSync(require.resolve('./components/voice_capture/voice.js'), 'utf8'), sandbox);
    doc.createElement = create;
    return {handlers, elements, render(consent) { handlers.message({source:parent, data:{type:'streamlit:render',
      args:{scope:'one', hands_free_allowed:true, recognition_mode:'browser_default', browser_consent:consent}}}); }};
  }
  const first = mount(); first.render(false);
  const firstStart = first.elements.find(el => el.textContent === 'Start hands-free');
  firstStart.onclick(); await new Promise(r => setImmediate(r)); assert.equal(starts, 0);
  first.render(true); firstStart.onclick(); await new Promise(r => setImmediate(r)); assert.equal(starts, 1);
  const replacement = mount(); assert.equal(aborts, 1); assert.equal(listeners.click.size, 1);
  replacement.render(true); const start = replacement.elements.find(el => el.textContent === 'Start hands-free');
  await new Promise(r => setImmediate(r)); assert.equal(starts, 2);
  for (const fn of listeners.click) fn({target:{closest:selector => selector.includes('send_button')}});
  await new Promise(r => setImmediate(r));
  assert.equal(parent.__vaaniVoiceManager.enabled, true); assert.equal(starts, 3);
  doc.hidden = true; for (const fn of listeners.visibilitychange) fn();
  assert.equal(parent.__vaaniVoiceManager.enabled, false); doc.hidden = false;
  await new Promise(r => setImmediate(r)); assert.equal(starts, 3);
  start.onclick(); await new Promise(r => setImmediate(r)); assert.equal(starts, 4);
  parent.__vaaniVoiceManager.dispose(); assert.equal(listeners.click.size, 0);
});
test('wake matching is bounded and handles punctuation', () => {
  assert.ok(isWake('Hey Vaani.')); assert.ok(isWake('hey, vani'));
  assert.ok(isWake('hey bani')); assert.ok(isWake('hello wani'));
  assert.ok(isWake('hi vaani')); assert.ok(isWake('hi vani'));
  assert.ok(isWake('hello vaani')); assert.ok(isWake('Vaani'));
  assert.ok(isWake('vani')); assert.ok(isWake('vaanishield'));
  assert.ok(isWake('वानी')); assert.ok(isWake('वाणी')); assert.ok(isWake('ವಾಣಿ'));
  assert.ok(isWake('हे वाणी')); assert.ok(isWake('ಹೇ ವಾಣಿ'));
  assert.equal(wakeCommand('हೇ ವಾಣಿ, ನನ್ನ ಬ್ಯಾಲೆನ್ಸ್ ತೋರಿಸಿ'), 'ನನ್ನ ಬ್ಯಾಲೆನ್ಸ್ ತೋರಿಸಿ');
  assert.equal(isWake('hey vanilla'), false); assert.equal(isWake('vanilla'), false);
});
test('WAV is mono 16 kHz PCM and resampled with a valid header', () => {
  const out = wav([new Float32Array(48000).fill(0.5)], 48000), view = new DataView(out.buffer);
  assert.equal(Buffer.from(out.subarray(0,4)).toString(),'RIFF');
  assert.equal(view.getUint32(24,true),16000); assert.equal(out.length,32044);
  assert.equal(view.getUint16(22,true),1); assert.equal(view.getInt16(44,true),16383);
});
test('consent, local-only availability and wake trigger submit once', async () => {
  const {c, sent} = harness(); assert.equal(c.enabled,false); assert.equal(c.recognition,undefined);
  await c.enable(); const rec = c.recognition;
  assert.equal(rec.processLocally,true); assert.equal(c.state,'STANDBY');
  rec.onresult({resultIndex:0,results:[[{transcript:'Hey Vaani'}]]});
  await new Promise(r => setTimeout(r, 600));
  assert.equal(sent.length,1); assert.equal(sent[0].kind,'wake'); assert.equal(rec.aborted,true);
  c.stop();
});
test('unsupported recognition never starts a remote fallback', async () => {
  const {c} = harness({SR:class {}}); await c.enable();
  assert.equal(c.state,'UNAVAILABLE'); assert.equal(c.enabled,false);
});
test('permission denied releases resources and stays stopped', async () => {
  const {c} = harness({getUserMedia:async () => { throw new Error('denied'); }}); await c.record(); assert.equal(c.state,'ERROR'); assert.equal(c.enabled,false);
});
test('mute while permission is pending stops the eventual stream', async () => {
  let resolve, stopped = false;
  const {c} = harness({getUserMedia:() => new Promise(r => { resolve = r; })});
  const p = c.record(); c.stop(); resolve({getTracks:() => [{stop:() => { stopped = true; }}]});
  await p; assert.equal(stopped,true); assert.equal(c.state,'IDLE');
});
test('silence auto-submits bounded audio exactly once', () => {
  const {c,sent} = harness(); c.state='RECORDING'; c.rate=16000; c.samples=0; c.speechMs=0; c.silenceMs=0; c.automatic=true;
  c.capture(new Float32Array(4000).fill(0.1)); c.capture(new Float32Array(18000)); c.finish();
  assert.equal(sent.length,1); assert.equal(sent[0].kind,'audio'); assert.equal(c.state,'PROCESSING');
  assert.equal(c.chunks.length,0);
});
test('a recording keeps one event identity and a later recording gets a new identity', async () => {
  const h = harness(); await h.c.record(); const firstId = h.c.recordingId;
  h.c.capture(new Float32Array(4000).fill(0.1)); h.c.finish(); h.c.finish();
  assert.equal(h.sent.length, 1); assert.equal(h.sent[0].id, firstId);
  await h.c.record(); h.c.capture(new Float32Array(4000).fill(0.1)); h.c.finish();
  assert.equal(h.sent.length, 2); assert.notEqual(h.sent[1].id, firstId); h.c.stop();
});
test('stopped and superseded responses cannot play; replay delivery is deduplicated', async () => {
  const h = harness(); h.c.send({kind:'wake'}); h.c.stop(); await h.c.response({id:'1',audio:'old'});
  assert.equal(h.plays(),0);
  h.c.send({kind:'audio'}); const p = h.c.response({id:'2',audio:'new'});
  assert.equal(h.plays(),1); await h.c.response({id:'2',audio:'new'}); assert.equal(h.plays(),1);
  h.c.stop(); await p; assert.equal(h.c.state,'IDLE');
});
test('wake during TTS stops playback before starting a new capture', async () => {
  const h = harness(); await h.c.enable(); h.c.send({kind:'audio'});
  const p = h.c.response({id:'1',audio:'tts'});
  await new Promise(r => setTimeout(r, 0));
  let called = false; h.c.record = async automatic => { assert.equal(automatic,true); called=true; };
  h.c.recognition.onresult({resultIndex:0,results:[[{transcript:'Hey Vaani'}]]});
  await new Promise(r => setTimeout(r, 600));
  await p; assert.equal(called,true); assert.ok(h.stops() >= 2); h.c.stop();
});
test('same-phrase commands preserve buffered audio and never send recognized text', async () => {
  for (const phrase of ['Hey Vaani, show my balance', 'HEY, VAANI! Show my balance', 'hey vani: show my balance']) {
    const h = harness(); await h.c.enable();
    h.c.captureWake(new Float32Array(4000).fill(0.1));
    const rec = h.c.recognition, event = {resultIndex:0,results:[[{transcript:phrase}]]};
    rec.onresult(event); rec.onresult(event);
    await new Promise(r => setTimeout(r, 300));
    assert.equal(h.sent.length, 1); assert.equal(h.sent[0].kind, 'audio');
    assert.equal(Buffer.from(h.sent[0].audio, 'base64').subarray(0,4).toString(), 'RIFF');
    assert.equal('text' in h.sent[0], false); h.c.stop();
  }
});
test('command in a later recognition result uses buffered audio once', async () => {
  const h = harness(); await h.c.enable(); h.c.captureWake(new Float32Array(4000).fill(0.1));
  const rec = h.c.recognition;
  rec.onresult({resultIndex:0,results:[[{transcript:'Hey Vaani'}]]});
  rec.onresult({resultIndex:1,results:[[{transcript:'Hey Vaani'}],[{transcript:'show my balance'}]]});
  await new Promise(r => setTimeout(r, 600));
  assert.equal(h.sent.length, 1); assert.equal(h.sent[0].kind, 'audio'); h.c.stop();
});
test('wake microphone denial and recognition initialization failure are actionable', async () => {
  const denied = harness({getUserMedia:async () => { throw Object.assign(new Error(), {name:'NotAllowedError'}); }});
  await denied.c.enable(); assert.equal(denied.c.enabled, false); assert.match(denied.states.at(-1)[1], /permission denied/i);
  class Broken { static available = async () => 'available'; constructor() { throw new Error(); } }
  Broken.prototype.processLocally = false;
  const broken = harness({SR:Broken}); await broken.c.enable(); assert.equal(broken.c.enabled, false);
  assert.match(broken.states.at(-1)[1], /initialize/i);
});
test('stopping, restarting and a hidden tab cannot retain listeners or audio', async () => {
  let hidden = false; const h = harness({hidden:() => hidden});
  await h.c.enable(); const old = h.c.recognition; h.c.stop(); assert.equal(old.aborted, true);
  await h.c.enable(); assert.notEqual(h.c.recognition, old);
  h.c.captureWake(new Float32Array(100).fill(0.1)); hidden = true; h.c.stop();
  assert.equal(h.c.enabled, false); assert.equal(h.c.wakeChunks.length, 0); assert.equal(h.c.recognition, null);
});
test('cancel while buffered command is pending suppresses upload and stale playback', async () => {
  const h = harness(); await h.c.enable(); h.c.captureWake(new Float32Array(4000).fill(0.1));
  h.c.recognition.onresult({resultIndex:0,results:[[{transcript:'Hey Vaani, balance'}]]});
  h.c.stop(); await new Promise(r => setTimeout(r, 300));
  assert.equal(h.sent.length, 0); await h.c.response({id:'old',audio:'tts'}); assert.equal(h.plays(), 0);
});
test('after normal voice playback unsolicited audio cannot start a second command', async () => {
  const h = harness(); await h.c.enable(); h.c.send({kind:'audio'});
  let captured=0; h.c.record=async () => { captured++; };
  const p=h.c.response({id:'1',audio:'tts'}); h.finishPlayback(); await p;
  await new Promise(r => setImmediate(r)); assert.equal(captured,0);
  assert.equal(h.c.state, 'STANDBY');
  h.c.send({kind:'audio'}); await h.c.response({id:'2',error:true});
  assert.equal(h.c.state,'ERROR'); assert.equal(captured,0); h.c.stop();
});
test('wake-only greeting still starts a command capture', async () => {
  const h = harness(); await h.c.enable(); h.c.send({kind:'wake'});
  let captured=0; h.c.record=async automatic => { assert.equal(automatic,true); captured++; };
  const p=h.c.response({id:'1',audio:'tts'}); h.finishPlayback(); await p;
  assert.equal(captured,1); h.c.stop();
});
test('push-to-talk answer returns to standby; a new manual recording still works', async () => {
  const h = harness(); await h.c.enable(); await h.c.record();
  h.c.capture(new Float32Array(4000).fill(0.1)); h.c.finish(); h.c.finish();
  const p=h.c.response({id:h.sent[0].id,audio:'tts'}); h.finishPlayback(); await p;
  await new Promise(r => setImmediate(r));
  h.c.capture(new Float32Array(4000).fill(0.1)); h.c.capture(new Float32Array(18000));
  assert.equal(h.sent.length,1); assert.equal(h.c.state,'STANDBY');
  await h.c.record(); h.c.capture(new Float32Array(4000).fill(0.1)); h.c.finish();
  assert.equal(h.sent.length,2); assert.notEqual(h.sent[0].id,h.sent[1].id); h.c.stop();
});
test('playback ending during microphone startup creates only one wake recognizer', async () => {
  let starts = 0, resolveMic;
  class SR { static available=async () => 'available'; start(){ starts++; } abort(){} }
  SR.prototype.processLocally = false;
  const h = harness({SR}); await h.c.enable(); h.c.interrupt();
  h.env.getUserMedia = () => new Promise(resolve => { resolveMic = resolve; });
  h.c.send({kind:'audio'}); const p=h.c.response({id:'1',audio:'tts'});
  h.finishPlayback(); await p;
  resolveMic({getTracks:() => [{stop(){}}]}); await new Promise(r => setImmediate(r));
  assert.equal(starts,2); assert.ok(h.c.recognition); assert.equal(h.c.state,'STANDBY'); h.c.stop();
});
test('interim and final results use en-IN and never emit duplicate wake or command events', async () => {
  const h = harness(); await h.c.enable('browser_default', true);
  const rec = h.c.recognition;
  assert.equal(rec.lang, 'en-IN'); assert.equal(rec.continuous, true); assert.equal(rec.interimResults, true);
  const interim = Object.assign([{transcript:'hey vaani'}], {isFinal:false});
  const final = Object.assign([{transcript:'Hey Vaani.'}], {isFinal:true});
  rec.onresult({resultIndex:0, results:[interim]});
  rec.onresult({resultIndex:0, results:[final]});
  await new Promise(r => setTimeout(r, 600));
  assert.equal(h.sent.length, 1); assert.equal(h.sent[0].kind, 'wake');
  const stages = h.traces.map(t => t.stage);
  assert.equal(stages.filter(s => s === 'wake_phrase_detected').length, 1);
  assert.ok(stages.includes('duplicate_event_rejected'));
  h.c.stop();
});
test('normal silence no-speech ends restart standby repeatedly without hitting quickEnds stop', async () => {
  const h = harness(); await h.c.enable('browser_default', true);
  for (let i = 0; i < 4; i++) {
    const rec = h.c.recognition;
    assert.ok(rec);
    rec.onerror({error:'no-speech'});
    rec.onend();
    await new Promise(r => setTimeout(r, 540));
    assert.equal(h.c.enabled, true);
    assert.equal(h.c.state, 'STANDBY');
  }
  h.c.stop();
});
test('privacy-safe diagnostics distinguish duplicate callback from two separate recordings', async () => {
  const h = harness();
  await h.c.record(false);
  const firstId = h.c.recordingId;
  h.c.capture(new Float32Array(4000).fill(0.1));
  h.c.finish();
  h.c.finish(); // duplicate finish on first recording
  const playPromise = h.c.response({id:firstId, audio:'tts'});
  h.finishPlayback();
  await playPromise;
  await h.c.response({id:firstId, audio:'tts'}); // duplicate response delivery
  await h.c.record(false);
  const secondId = h.c.recordingId;
  h.c.capture(new Float32Array(4000).fill(0.1));
  h.c.finish();
  assert.notEqual(firstId, secondId);
  assert.equal(h.sent.length, 2);
  const stagesForFirst = h.traces.filter(t => t.event_id === firstId).map(t => t.stage);
  assert.deepEqual(stagesForFirst, [
    'push_to_talk_recording_started',
    'push_to_talk_recording_stopped',
    'audio_event_created',
    'component_event_sent',
    'duplicate_event_rejected',
    'response_received',
    'response_playback_started',
    'response_playback_completed',
    'duplicate_event_rejected',
  ]);
  const stagesForSecond = h.traces.filter(t => t.event_id === secondId).map(t => t.stage);
  assert.deepEqual(stagesForSecond, [
    'push_to_talk_recording_started',
    'push_to_talk_recording_stopped',
    'audio_event_created',
    'component_event_sent',
  ]);
  for (const entry of h.traces) {
    assert.equal('transcript' in entry, false);
    assert.equal('audio' in entry, false);
  }
  h.c.stop();
});
