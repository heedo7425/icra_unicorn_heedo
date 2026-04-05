#!/usr/bin/env python3
"""
PCD → Interactive HTML Z-Layer Viewer 생성기
사용법: python3 gen_pcd_viewer.py [pcd_path] [waypoints_json_path] [output_html_path]
       인자 없으면 기본값(eng_0404_v2) 사용

생성된 HTML은 단독 파일 — 브라우저에서 열면 바로 동작 (서버 불필요)
PCD 포인트가 전부 JSON으로 HTML 안에 embedded됨
"""
import numpy as np
import json
import sys
import os

# ============================================================
# 설정
# ============================================================
DEFAULT_PCD = '/home/unicorn/catkin_ws/src/race_stack/stack_master/maps/eng_0404_v2/eng_0404_v2.pcd'
DEFAULT_WP_JSON = '/home/unicorn/catkin_ws/src/race_stack/stack_master/maps/eng_0404_v2/global_waypoints.json'
DEFAULT_OUTPUT = '/home/unicorn/catkin_ws/src/race_stack/HJ_docs/debug/figure/0404/pcd_layers/pcd_z_slider.html'

# 트랙 영역 필터 (필요시 수정)
X_MIN, X_MAX = -12, 20
Y_MIN, Y_MAX = -12, 6
Z_MIN, Z_MAX = -0.5, 1.0

# ============================================================
# 인자 파싱
# ============================================================
pcd_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PCD
wp_json_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_WP_JSON
output_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT

# ============================================================
# 1. PCD 로드 (binary format)
# ============================================================
print(f"Loading PCD: {pcd_path}")
with open(pcd_path, 'rb') as f:
    n_points = 0
    fields = []
    data_type = 'binary'
    while True:
        line = f.readline().decode('ascii', errors='ignore').strip()
        if line.startswith('POINTS'):
            n_points = int(line.split()[1])
        elif line.startswith('FIELDS'):
            fields = line.split()[1:]
        elif line.startswith('DATA'):
            data_type = line.split()[1]
            break

    if data_type == 'binary':
        data = np.fromfile(f, dtype=np.float32).reshape(-1, len(fields))
    else:
        data = np.loadtxt(f, max_rows=n_points)

print(f"  Total points: {len(data)}, Fields: {fields}")
print(f"  X: {data[:,0].min():.2f}~{data[:,0].max():.2f}")
print(f"  Y: {data[:,1].min():.2f}~{data[:,1].max():.2f}")
print(f"  Z: {data[:,2].min():.2f}~{data[:,2].max():.2f}")

# 트랙 영역 필터
mask = ((data[:,0] > X_MIN) & (data[:,0] < X_MAX) &
        (data[:,1] > Y_MIN) & (data[:,1] < Y_MAX) &
        (data[:,2] > Z_MIN) & (data[:,2] < Z_MAX))
track = np.round(data[mask, :3], 4)
print(f"  After filter: {len(track)} points")

pts_json = json.dumps(track.tolist())

# ============================================================
# 2. Waypoints 로드
# ============================================================
wp_json = '[]'
if os.path.exists(wp_json_path):
    print(f"Loading waypoints: {wp_json_path}")
    d = json.load(open(wp_json_path))
    # global_waypoints.json 구조에서 웨이포인트 찾기
    wpnts = None
    for key in ['global_traj_wpnts_sp', 'global_traj_wpnts_iqp', 'centerline_waypoints']:
        if key in d and isinstance(d[key], dict) and 'wpnts' in d[key]:
            wpnts = d[key]['wpnts']
            print(f"  Using {key}: {len(wpnts)} waypoints")
            break
    if wpnts:
        wp_json = json.dumps([[round(w['x_m'],4), round(w['y_m'],4), round(w.get('z_m',0),4)] for w in wpnts])
else:
    print(f"  No waypoints file found, skipping")

# ============================================================
# 3. HTML 생성
# ============================================================
html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PCD Z-Layer Viewer</title>
<style>
  body {{ margin: 0; font-family: monospace; background: #1a1a1a; color: #eee; }}
  #controls {{ position: fixed; top: 10px; left: 10px; z-index: 10; background: rgba(0,0,0,0.85); padding: 15px; border-radius: 8px; width: 320px; }}
  #controls label {{ display: block; margin: 5px 0; }}
  #controls input[type=range] {{ width: 100%; }}
  #info {{ position: fixed; bottom: 10px; left: 10px; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 8px; font-size: 12px; }}
  canvas {{ display: block; }}
  .slider-row {{ display: flex; align-items: center; gap: 8px; }}
  .slider-row span {{ min-width: 60px; text-align: right; font-size: 13px; }}
</style>
</head>
<body>
<div id="controls">
  <h3 style="margin:0 0 10px">PCD Z-Layer Viewer</h3>
  <label>Z Center (cm):
    <div class="slider-row">
      <input type="range" id="zCenter" min="-50" max="100" value="-8" step="0.5">
      <span id="zCenterVal">-8</span>
    </div>
  </label>
  <label>Z Thickness (cm):
    <div class="slider-row">
      <input type="range" id="zThick" min="1" max="30" value="5" step="1">
      <span id="zThickVal">5</span>
    </div>
  </label>
  <label>Point Size:
    <div class="slider-row">
      <input type="range" id="ptSize" min="0.5" max="5" value="1.5" step="0.5">
      <span id="ptSizeVal">1.5</span>
    </div>
  </label>
  <label><input type="checkbox" id="showWp" checked> Show Waypoints</label>
  <div id="pointCount" style="margin-top:8px; font-size:12px;">Points: -</div>
</div>
<div id="info">Drag=pan, Scroll=zoom</div>
<canvas id="c"></canvas>
<script>
const P = {pts_json};
const W = {wp_json};
const c = document.getElementById('c');
const ctx = c.getContext('2d');
let ox=-80, oy=120, sc=40, drag=false, lx=0, ly=0;
function resize(){{c.width=innerWidth;c.height=innerHeight;draw();}}
addEventListener('resize',resize);
c.onmousedown=e=>{{drag=true;lx=e.clientX;ly=e.clientY;}};
c.onmousemove=e=>{{if(!drag)return;ox+=e.clientX-lx;oy+=e.clientY-ly;lx=e.clientX;ly=e.clientY;draw();}};
c.onmouseup=()=>drag=false;
c.onmouseleave=()=>drag=false;
c.onwheel=e=>{{e.preventDefault();const f=e.deltaY>0?0.9:1.1;ox=e.clientX-(e.clientX-ox)*f;oy=e.clientY-(e.clientY-oy)*f;sc*=f;draw();}};
const zCS=document.getElementById('zCenter'),zTS=document.getElementById('zThick'),pS=document.getElementById('ptSize'),wC=document.getElementById('showWp');
[zCS,zTS,pS].forEach(s=>s.oninput=draw);
wC.onchange=draw;
function w2s(wx,wy){{return[c.width/2+ox+wx*sc, c.height/2+oy-wy*sc];}}
function draw(){{
  const zc=+zCS.value/100, zt=+zTS.value/100, ps=+pS.value;
  document.getElementById('zCenterVal').textContent=zCS.value;
  document.getElementById('zThickVal').textContent=zTS.value;
  document.getElementById('ptSizeVal').textContent=pS.value;
  const zlo=zc-zt/2, zhi=zc+zt/2;
  ctx.fillStyle='#1a1a1a';ctx.fillRect(0,0,c.width,c.height);
  ctx.strokeStyle='#333';ctx.lineWidth=0.5;
  for(let g=-20;g<=25;g+=5){{let[a,b]=w2s(g,-15),[d,e]=w2s(g,10);ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(d,e);ctx.stroke();}}
  for(let g=-15;g<=10;g+=5){{let[a,b]=w2s(-20,g),[d,e]=w2s(25,g);ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(d,e);ctx.stroke();}}
  let n=0;
  for(let i=0;i<P.length;i++){{
    const pz=P[i][2];
    if(pz<zlo||pz>=zhi)continue;
    const[sx,sy]=w2s(P[i][0],P[i][1]);
    if(sx<-5||sx>c.width+5||sy<-5||sy>c.height+5)continue;
    const t=(pz-zlo)/(zhi-zlo+1e-6);
    const r=Math.round(Math.min(255,Math.max(0,255*(1.5-Math.abs(t-0.75)*4))));
    const g=Math.round(Math.min(255,Math.max(0,255*(1.5-Math.abs(t-0.5)*4))));
    const b=Math.round(Math.min(255,Math.max(0,255*(1.5-Math.abs(t-0.25)*4))));
    ctx.fillStyle='rgb('+r+','+g+','+b+')';
    ctx.fillRect(sx-ps/2,sy-ps/2,ps,ps);
    n++;
  }}
  if(wC.checked&&W.length>0){{
    ctx.strokeStyle='#0f0';ctx.lineWidth=1.5;ctx.beginPath();
    for(let i=0;i<W.length;i++){{const[sx,sy]=w2s(W[i][0],W[i][1]);i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}}
    ctx.closePath();ctx.stroke();
  }}
  document.getElementById('pointCount').textContent='Points: '+n+' | Z: '+(zlo*100).toFixed(1)+'~'+(zhi*100).toFixed(1)+'cm';
}}
resize();
</script>
</body></html>'''

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    f.write(html)

print(f"\nOutput: {output_path} ({len(html)//1024}KB)")
print(f"Points embedded: {len(track)}")
print("Open in browser — no server needed!")
