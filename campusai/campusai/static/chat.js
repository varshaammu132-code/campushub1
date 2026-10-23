const $=id=>document.getElementById(id);
$('bubble').onclick=()=>{$('panel').hidden=!$('panel').hidden};
function add(cls,text,sources){const d=document.createElement('div');d.className=cls;d.textContent=text; // textContent only: no innerHTML
 (sources||[]).forEach(u=>{if(!/^https?:\/\//.test(u))return;const a=document.createElement('a');a.href=u;a.textContent=u;a.target='_blank';a.rel='noopener noreferrer';d.append(document.createElement('br'),a)});
 $('log').appendChild(d)}
$('send').onclick=async()=>{const q=$('q').value.trim();if(!q)return;add('user',q);$('q').value='';$('err').textContent='';$('send').disabled=true;
 try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  if(!r.ok)throw new Error(r.status==429?'Too many requests — please wait a moment.':'Something went wrong. Please retry.');
  const d=await r.json();add('bot',d.answer,d.sources)}catch(e){$('err').textContent=e.message}finally{$('send').disabled=false}};
