import React, { useEffect, useMemo, useState } from 'react'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, BarChart, Bar
} from 'recharts'

const API = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

const nav = [
  ['overview', '01', 'Mission Overview'],
  ['demo', '02', 'Mission Video Demo'],
  ['analytics', '03', 'Flight Analytics'],
  ['audit', '04', 'Evidence Auditor'],
  ['copilot', '05', 'AI Mission Engineer'],
  ['architecture', '06', 'System Architecture']
]

function Metric({label, value, note}) {
  return <div className="metric panel">
    <div className="eyebrow">{label}</div>
    <div className="metric-value">{value ?? '—'}</div>
    <div className="muted small">{note}</div>
  </div>
}

function Status({online}) {
  return <span className={'status ' + (online ? 'online' : 'offline')}>
    <i /> {online ? 'API CONNECTED' : 'API DISCONNECTED'}
  </span>
}

function formatValue(v) {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return Number(v.toFixed(4)).toString()
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function MissionVideoPanel() {
  const videoUrl = '/videos/validated_c18_airborne_demo_h264.mp4'
  return <div className="panel demo-panel">
    <div className="section-heading"><span>VALIDATED C18 / AIRBORNE RUN</span><span>RECORDED EXPERIMENT</span></div>
    <video className="mission-video" controls playsInline preload="metadata">
      <source src={videoUrl} type="video/mp4"/>
      Your browser does not support HTML5 video.
    </video>
    <div className="demo-caption">
      <div><div className="eyebrow green">AI-IN-THE-LOOP TRACKING</div><h2>Airborne optical beacon tracking</h2></div>
      <span className="demo-format">MP4 · H.264</span>
    </div>
    <p className="muted small">Recorded simulation demonstration — not a live camera feed or live aircraft connection.</p>
  </div>
}

function Overview({data, audit, online}) {
  const metrics = data?.metrics || {}
  const centroid = metrics.centroid_error_px || {}
  const confidence = metrics.confidence || {}
  const latency = metrics.inference_latency_ms || {}

  return <div className="page">
    <div className="hero panel">
      <div>
        <div className="eyebrow green">MISSION CONTROL / FSOC-01</div>
        <h1>OPTICAL TRACKING<br/><span>FLIGHT DECK</span></h1>
        <p>AI-based virtual camera tracking for coarse alignment of mobile free-space optical communication terminals.</p>
        <div className="hero-tags">
          <span>RECORDED EXPERIMENTS</span><span>VIRTUAL CAMERA</span><span>AI-ASSISTED ANALYSIS</span>
        </div>
      </div>
      <Radar />
    </div>

    <MissionVideoPanel/>

    <div className="section-heading"><span>MISSION TELEMETRY</span><span>RECORDED DATA / NOT LIVE SENSORS</span></div>
    <div className="metrics-grid">
      <Metric label="CENTROID MEAN ERROR" value={centroid.mean != null ? `${formatValue(centroid.mean)} px` : '—'} note="Detector centroid error"/>
      <Metric label="CENTROID RMSE" value={centroid.rmse != null ? `${formatValue(centroid.rmse)} px` : '—'} note="Recorded detector benchmark"/>
      <Metric label="VALID FRAMES" value={centroid.valid_frames ?? data?.frame_rows ?? '—'} note="C18 recorded flight run"/>
      <Metric label="MEAN CONFIDENCE" value={confidence.mean != null ? formatValue(confidence.mean) : '—'} note="Detector confidence"/>
      <Metric label="INFERENCE LATENCY" value={latency.mean != null ? `${formatValue(latency.mean)} ms` : '—'} note="Recorded detector latency"/>
      <Metric label="AUDIT FINDINGS" value={audit?.finding_count ?? '—'} note="Deterministic consistency checks"/>
    </div>

    <div className="grid-two">
      <div className="panel section-panel">
        <div className="section-heading"><span>SYSTEM STATUS</span><span>01 / STATUS</span></div>
        <div className="status-row"><span>Backend API</span><Status online={online}/></div>
        <div className="status-row"><span>Evidence report</span><b>{data ? 'LOADED' : 'PENDING'}</b></div>
        <div className="status-row"><span>Deterministic audit</span><b>{audit ? 'EXECUTED' : 'PENDING'}</b></div>
        <div className="status-row"><span>AI copilot</span><b>MOCK / NEBIUS</b></div>
      </div>
      <div className="panel section-panel">
        <div className="section-heading"><span>MISSION NOTE</span><span>02 / CAUTION</span></div>
        <p className="muted">This interface presents saved experimental evidence. It does not indicate a live aircraft connection, live optical terminal, or active sensor feed.</p>
        <p className="warning-text">Detector centroid error and closed-loop tracking error are different measurements. Interpret them separately.</p>
      </div>
    </div>
  </div>
}

function MissionDemo() {
  return <div className="page">
    <PageTitle code="02 / RECORDED FLIGHT DEMONSTRATION" title="MISSION VIDEO DEMO" subtitle="Playback of the existing validated C18 airborne AI-in-the-loop FSOC tracking experiment."/>
    <MissionVideoPanel/>
  </div>
}

function Radar() {
  return <div className="radar-wrap">
    <div className="radar">
      <div className="radar-sweep"/>
      <div className="radar-cross"/>
      <span className="radar-dot d1"/><span className="radar-dot d2"/><span className="radar-dot d3"/>
      <span className="radar-label">VIRTUAL FPA</span>
    </div>
    <div className="radar-caption">SIMULATION DISPLAY / NO LIVE TARGET</div>
  </div>
}

function Analytics({data}) {
  const rows = data?.trajectory_records || []
  const chartData = useMemo(() => rows.map(r => ({
    name: String(r.Trajectory ?? r.trajectory ?? 'Unknown'),
    rmse: Number(r['RMSE (px)'] ?? r.rmse ?? 0),
    max: Number(r['Maximum Error (px)'] ?? r.maximum_error ?? 0),
    lock: Number(r['≤10 px Lock (%)'] ?? r['Lock (%)'] ?? r.lock_percent ?? 0)
  })), [rows])

  return <div className="page">
    <PageTitle code="02 / ANALYTICS" title="FLIGHT ANALYTICS" subtitle="Recorded trajectory validation and performance evidence."/>
    <div className="panel chart-panel">
      <div className="section-heading"><span>TRAJECTORY RMSE</span><span>PIXELS</span></div>
      {chartData.length ? <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}><CartesianGrid stroke="#183126" strokeDasharray="3 3"/>
          <XAxis dataKey="name" stroke="#8ba99b"/><YAxis stroke="#8ba99b"/>
          <Tooltip contentStyle={{background:'#07130e',border:'1px solid #24533d'}}/>
          <Bar dataKey="rmse" fill="#00e89a" radius={[4,4,0,0]}/>
        </BarChart>
      </ResponsiveContainer> : <Empty/>}
    </div>
    <div className="panel chart-panel">
      <div className="section-heading"><span>MAXIMUM TRACKING ERROR</span><span>PIXELS</span></div>
      {chartData.length ? <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData}><CartesianGrid stroke="#183126" strokeDasharray="3 3"/>
          <XAxis dataKey="name" stroke="#8ba99b"/><YAxis stroke="#8ba99b"/>
          <Tooltip contentStyle={{background:'#07130e',border:'1px solid #24533d'}}/>
          <Line type="monotone" dataKey="max" stroke="#ffc857" strokeWidth={2}/>
        </LineChart>
      </ResponsiveContainer> : <Empty/>}
    </div>
    <div className="panel table-panel">
      <div className="section-heading"><span>TRAJECTORY RECORDS</span><span>{rows.length} RECORDS</span></div>
      <div className="table-scroll"><table><thead><tr><th>TRAJECTORY</th><th>RMSE (PX)</th><th>MAX ERROR (PX)</th><th>LOCK</th></tr></thead>
      <tbody>{chartData.map((r,i)=><tr key={i}><td>{r.name}</td><td>{formatValue(r.rmse)}</td><td>{formatValue(r.max)}</td><td>{formatValue(r.lock)}%</td></tr>)}</tbody></table></div>
    </div>
    <p className="muted small">Trajectory values are historical recorded results. Unusually small RMSE values should be checked against raw logs.</p>
  </div>
}

function Audit({audit}) {
  const findings = audit?.findings || []
  return <div className="page">
    <PageTitle code="03 / INTEGRITY" title="EVIDENCE AUDITOR" subtitle="Read-only deterministic consistency checks against saved source files."/>
    <div className="metrics-grid">
      <Metric label="TOTAL FINDINGS" value={audit?.finding_count ?? '—'} note="Checks and review flags"/>
      <Metric label="PASS" value={findings.filter(x=>x.severity==='PASS').length} note="Specific checks passed"/>
      <Metric label="REVIEW" value={findings.filter(x=>x.severity==='REVIEW').length} note="Require investigation"/>
    </div>
    <div className="panel section-panel">
      <div className="section-heading"><span>AUDIT FINDINGS</span><span>READ ONLY</span></div>
      {findings.map((f,i)=><div className="finding" key={i}>
        <div><span className={'severity '+(f.severity==='PASS'?'pass':'review')}>{f.severity}</span><b>{f.code}</b></div>
        <p>{f.message}</p>
        {f.details && Object.keys(f.details).length>0 && <details><summary>Inspect details</summary><pre>{JSON.stringify(f.details,null,2)}</pre></details>}
      </div>)}
      {!findings.length && <Empty/>}
    </div>
    <p className="muted small">PASS applies only to the particular consistency check performed. REVIEW indicates further investigation, not proof of an error.</p>
  </div>
}

function Copilot() {
  const [question,setQuestion] = useState('')
  const [mode,setMode] = useState('mock')
  const [answer,setAnswer] = useState(null)
  const [busy,setBusy] = useState(false)
  const [error,setError] = useState('')

  async function ask(e) {
    e.preventDefault()
    if (!question.trim()) return
    setBusy(true); setError(''); setAnswer(null)
    try {
      const r = await fetch(`${API}/api/copilot`,{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({question,mode})
      })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail || 'Copilot request failed')
      setAnswer(body)
    } catch(err) { setError(err.message) }
    finally { setBusy(false) }
  }

  return <div className="page">
    <PageTitle code="04 / AI SYSTEMS" title="AI MISSION ENGINEER" subtitle="Ask engineering questions grounded in the project's saved evidence."/>
    <div className="panel copilot-panel">
      <div className="copilot-top"><span className="eyebrow green">SKYTRACK INTELLIGENCE</span><span className="small muted">EVIDENCE-AWARE ASSISTANT</span></div>
      <form onSubmit={ask}>
        <label>MODEL MODE</label>
        <select value={mode} onChange={e=>setMode(e.target.value)}>
          <option value="mock">LOCAL MOCK / DETERMINISTIC</option>
          <option value="nebius">NEBIUS / NVIDIA MODEL</option>
        </select>
        <label>ENGINEERING QUERY</label>
        <textarea value={question} onChange={e=>setQuestion(e.target.value)}
          placeholder="Example: Summarize the trajectory validation results and identify evidence that requires further review."
          rows={5} maxLength={4000}/>
        <button className="primary" disabled={busy || !question.trim()}>{busy?'ANALYZING...':'RUN MISSION ANALYSIS →'}</button>
      </form>
      {error && <div className="error-box">{error}</div>}
      {answer && <div className="answer">
        <div className="section-heading"><span>ANALYSIS OUTPUT</span><span>{answer.mode?.toUpperCase()}</span></div>
        <pre>{answer.answer}</pre>
        {answer.warning && <p className="warning-text">{answer.warning}</p>}
        <details><summary>Evidence audit findings</summary><pre>{JSON.stringify(answer.audit_findings,null,2)}</pre></details>
      </div>}
    </div>
    <div className="panel section-panel">
      <div className="eyebrow green">SUGGESTED QUERIES</div>
      {['Summarize the recorded detector performance.','Which trajectory results require validation?','Explain the difference between centroid error and closed-loop tracking error.'].map(q=>
        <button className="suggestion" key={q} onClick={()=>setQuestion(q)}>{q} ↗</button>
      )}
    </div>
  </div>
}

function Architecture() {
  return <div className="page">
    <PageTitle code="05 / SYSTEM DESIGN" title="SYSTEM ARCHITECTURE" subtitle="Deployment architecture for the FSOC SkyTrack AI platform."/>
    <div className="architecture">
      <div className="arch-node"><span>01 / FRONTEND</span><h2>REACT + VITE</h2><p>Flight Deck interface · telemetry · analytics · AI workspace</p><small>VERCEL DEPLOYMENT TARGET</small></div>
      <div className="arch-arrow">↓ REST API / JSON ↓</div>
      <div className="arch-node"><span>02 / BACKEND</span><h2>PYTHON FASTAPI</h2><p>Health · overview · audit · copilot · architecture asset</p><small>RAILWAY DEPLOYMENT TARGET</small></div>
      <div className="arch-arrow">↓ INTERNAL MODULES ↓</div>
      <div className="arch-node"><span>03 / INTELLIGENCE</span><h2>AI + EVIDENCE ENGINE</h2><p>Copilot · deterministic auditor · saved experiment data</p><small>NEBIUS CONFIGURATION OPTIONAL</small></div>
    </div>
    <p className="muted small">Deployment targets are planned architecture, not confirmation that the services are already deployed.</p>
  </div>
}

function PageTitle({code,title,subtitle}) {
  return <div className="page-title"><div className="eyebrow green">{code}</div><h1>{title}</h1><p>{subtitle}</p></div>
}

function Empty() {
  return <div className="empty">NO DATA AVAILABLE / CHECK API CONNECTION</div>
}

export default function App() {
  const [active,setActive] = useState('overview')
  const [online,setOnline] = useState(false)
  const [data,setData] = useState(null)
  const [audit,setAudit] = useState(null)
  const [error,setError] = useState('')

  useEffect(()=>{
    let alive = true
    async function load() {
      try {
        const [h,o,a] = await Promise.all([
          fetch(`${API}/health`),
          fetch(`${API}/api/overview`),
          fetch(`${API}/api/audit`)
        ])
        if (!h.ok || !o.ok || !a.ok) throw new Error('Backend returned an error.')
        const [health,overview,auditData] = await Promise.all([h.json(),o.json(),a.json()])
        if (alive) {
          setOnline(health.status === 'healthy')
          setData(overview)
          setAudit(auditData)
          setError('')
        }
      } catch(e) {
        if (alive) {setOnline(false);setError(e.message)}
      }
    }
    load()
    return ()=>{alive=false}
  },[])

  const page = {
    overview:<Overview data={data} audit={audit} online={online}/>,
    demo:<MissionDemo/>,
    analytics:<Analytics data={data}/>,
    audit:<Audit audit={audit}/>,
    copilot:<Copilot/>,
    architecture:<Architecture/>
  }[active]

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">S<span>✦</span></div><div><b>SKYTRACK</b><small>FLIGHT DECK / AI</small></div></div>
      <div className="side-label">MISSION SYSTEMS</div>
      <nav>{nav.map(([id,num,label])=><button key={id} className={active===id?'nav-item active':'nav-item'} onClick={()=>setActive(id)}><span>{num}</span>{label}</button>)}</nav>
      <div className="side-bottom"><Status online={online}/><div className="small muted">FSOC RESEARCH PROTOTYPE<br/>BUILD 1.0.0</div></div>
    </aside>
    <main className="main">
      <header className="topbar"><div><span className="eyebrow">FSOC / VIRTUAL CAMERA TRACKING</span><span className="topbar-sub">MISSION CONTROL INTERFACE</span></div><div className="top-right"><Status online={online}/><span className="clock-label">RECORDED DATA MODE</span></div></header>
      {error && <div className="connection-error">BACKEND CONNECTION FAILED: {error}. Configure VITE_API_URL.</div>}
      {page}
      <footer>SKYTRACK // FLIGHT DECK <span>ENGINEERING RESEARCH INTERFACE · NOT A LIVE FLIGHT CONTROL SYSTEM</span></footer>
    </main>
  </div>
}
