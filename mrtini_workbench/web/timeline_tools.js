/* Pure timeline geometry helpers: no DOM or physical simulator state. */
'use strict';
(function(root){
 const duration=e=>e.channel==='adc'?e.samples*e.dwell_us/1000:e.duration_ms;
 const rounded=x=>Math.round(x*1e9)/1e9;
 const raster=x=>rounded(Math.round(x*100)/100);
 function snap(events,index,start,dur,{resize=false,tolerance=0,enabled=true,maxTime=Infinity}={}){
   const a=events[index],min=a.channel==='adc'?a.dwell_us/1000:.01;
   let value=resize?start+Math.max(min,dur):Math.max(0,start),best=null;
   const candidates=[{time:0,label:'Sequence start'},{time:maxTime,label:'TR end'}];
   events.forEach((e,i)=>{if(i===index)return;for(const [t,edge] of [[e.start_ms,'start'],[e.start_ms+duration(e),'end']])candidates.push({time:t,label:e.channel.toUpperCase()+' '+edge,same:e.channel===a.channel})});
   if(enabled)for(const c of candidates){if(!Number.isFinite(c.time))continue;for(const offset of resize?[0]:[0,dur]){
     const proposed=c.time-offset;if(proposed<0||resize&&proposed<start+min-1e-9)continue;
     const dist=Math.abs(value+offset-c.time);if(dist>tolerance)continue;
     if(!best||dist<best.dist-1e-9||(Math.abs(dist-best.dist)<1e-9&&c.same&&!best.same))best={...c,dist,proposed};
   }}
   let startOut=resize?start:Math.max(0,best?rounded(best.proposed):raster(value)),durOut=resize?Math.max(min,best?rounded(best.proposed-start):raster(dur)):dur;
   if(resize){if(a.channel==='adc')durOut=Math.max(1,Math.round(durOut/min))*min;else if(a.channel==='rf')durOut=Math.max(.02,raster(durOut));}
   // RF duration and ADC sample raster cannot always reach a foreign event boundary exactly.
   if(best&&resize&&Math.abs(startOut+durOut-best.time)>1e-8)best=null;
   return {start:rounded(startOut),duration:rounded(durOut),target:best?best.time:null,label:best?best.label:''};
 }
 function regenerate(events,n,fov,dwellUs){
   const read=events.find(e=>e.encoding_role==='adc');if(!read)return false;
   if(!Number.isInteger(n)||n<1||!Number.isFinite(fov)||fov<=0||!Number.isFinite(dwellUs)||dwellUs<=0)throw Error('Invalid encoding geometry');
   const roles=['gx-pre','gy-pre','gx-read','adc','gx-post','gy-post'];if(roles.some(r=>events.filter(e=>e.encoding_role===r).length!==1))throw Error('Incomplete generated encoding group. Use manual timing or restore the recipe.');
   const dk=1000/fov,dwell=dwellUs/1000,centre=read.start_ms+(read.samples/2+.5)*read.dwell_us/1000;
   const start=centre-(n/2+.5)*dwell,end=start+n*dwell;
   if(start<1)throw Error('Readout no longer fits before its centre; move it or increase timing.');
   const updates={
     'gx-pre':{start_ms:start-1,duration_ms:1,moment:-(n/2+.5)*dk,shot_step:0},
     'gy-pre':{start_ms:start-1,duration_ms:1,moment:-n/2*dk,shot_step:dk},
     'gx-read':{start_ms:start,duration_ms:n*dwell,moment:n*dk,shot_step:0},
     'adc':{start_ms:start,samples:n,dwell_us:dwellUs},
     'gx-post':{start_ms:end,duration_ms:1,moment:-(n/2-.5)*dk,shot_step:0},
     'gy-post':{start_ms:end,duration_ms:1,moment:n/2*dk,shot_step:-dk}
   };
   for(const e of events){if(updates[e.encoding_role])Object.assign(e,updates[e.encoding_role]);}return true;
 }
 const api={duration,snap,regenerate};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.MRtiniTimeline=api;
})(typeof window!=='undefined'?window:globalThis);
