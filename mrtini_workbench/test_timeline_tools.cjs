const assert=require('node:assert/strict');const {snap,regenerate}=require('./web/timeline_tools.js');
let ev=[{channel:'gx',start_ms:2,duration_ms:1},{channel:'gx',start_ms:4,duration_ms:2},{channel:'gy',start_ms:7,duration_ms:1}];
assert.equal(snap(ev,1,3.05,2,{tolerance:.1}).start,3);
assert.equal(snap(ev,0,2.95,1,{tolerance:.1}).start,3);
assert.equal(snap(ev,1,3.05,2,{tolerance:.1,enabled:false}).start,3.05);
assert.equal(snap(ev,1,4,2.95,{resize:true,tolerance:.1}).duration,3);
assert.equal(snap(ev,1,5.03,2,{tolerance:.1}).target,7);
let roles=['gx-pre','gy-pre','gx-read','adc','gx-post','gy-post'];ev=roles.map(r=>({encoding_role:r,channel:r==='adc'?'adc':r.slice(0,2),start_ms:5,duration_ms:1,samples:128,dwell_us:12}));
const centre=5+64.5*.012;regenerate(ev,64,200,20);let adc=ev[3];assert.equal(adc.samples,64);assert.equal(adc.dwell_us,20);assert(Math.abs(adc.start_ms+32.5*.02-centre)<1e-12);assert.equal(ev[2].moment,320);assert.equal(ev[1].shot_step,5);assert.throws(()=>regenerate(ev.slice(1),64,200,20),/Incomplete/);console.log('timeline helpers: 11 assertions passed');
