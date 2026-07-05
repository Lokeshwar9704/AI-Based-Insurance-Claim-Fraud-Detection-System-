import { useState, useEffect, useRef } from "react";

// ── Stable helper components — defined at module level so React never
//    creates new component types on re-render (critical for input focus)
function LB({ c, mb = 6, children }) {
  return (
    <div style={{ fontSize:10, fontFamily:"var(--di)", fontWeight:800, color:c||"var(--sub)", letterSpacing:".14em", textTransform:"uppercase", marginBottom:mb }}>
      {children}
    </div>
  );
}

function FD({ label, err, ext, children }) {
  return (
    <div style={{ marginBottom:12 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
        <LB mb={0}>{label}</LB>
        {ext && <span style={{ fontSize:9, fontFamily:"var(--di)", fontWeight:800, color:"var(--gn)", letterSpacing:".1em" }}>EXTRACTED</span>}
      </div>
      {children}
      {err && <div style={{ fontSize:10, color:"var(--rd)", marginTop:3, fontFamily:"var(--mo)", fontWeight:600 }}>{err}</div>}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
//  FRAUDGUARD v3 — Insurance Claim Fraud Detection
//  ML:  RF (55%) + ExtraTrees (15%) + GradientBoost (30%) Ensemble
//  AI:  Llama 3 70B via Groq API — 5-Agent Pipeline
//  UI:  Light theme · Syne + IBM Plex Mono fonts · Bold type system
// ═══════════════════════════════════════════════════════════════════════

// ── Trained EDA rates (fraud_oracle.csv, 15,420 rows) ────────────────
const EDA = {
  bp:   { Liability: 0.007, Collision: 0.073, "All Perils": 0.102 },
  ft:   { "Policy Holder": 0.079, "Third Party": 0.009 },
  pt:   { "Sport - Collision": 0.138, "Utility - All Perils": 0.121, "Sedan - All Perils": 0.101, "Utility - Collision": 0.082, "Sedan - Collision": 0.074, "Sedan - Liability": 0.007, "Sport - Liability": 0.000 },
  vc:   { Utility: 0.113, Sedan: 0.082, Sport: 0.016 },
  ac:   { "under 6 months": 0.750, "2 to 3 years": 0.175, "1 year": 0.065, "no change": 0.058, "4 to 8 years": 0.052 },
  aph:  { "16 to 17": 0.097, "18 to 20": 0.133, "21 to 25": 0.148, "26 to 30": 0.072, "31 to 35": 0.058, "36 to 40": 0.057, "41 to 50": 0.051, "51 to 65": 0.050 },
};

const FEAT_IMP = [
  { name: "injury_ratio", v: 0.2879 }, { name: "quartet_signal", v: 0.1117 },
  { name: "high_injury_ratio", v: 0.0443 }, { name: "no_police_report", v: 0.0377 },
  { name: "composite_score", v: 0.0331 }, { name: "fault_no_police", v: 0.0298 },
  { name: "address_change_rate", v: 0.0241 }, { name: "late_night", v: 0.0187 },
];

// ── ML scoring engine (mirrors trained ensemble) ──────────────────────
function mlScore(c) {
  const auth = c.authoritiesContacted || "Police";
  const wit  = Number(c.witnesses)    || 0;
  const mo   = Number(c.policyMonths) || 0;
  const prev = Number(c.prevClaims)   || 0;
  const inj  = Math.min(1, Math.max(0, Number(c.injuryRatio) || 0));
  const amt  = Number(c.claimAmount)  || 0;
  const hr   = Number(c.incidentHour) || 0;
  const bp   = c.basePolicy || "Collision";
  const fa   = c.fault || "Policy Holder";
  const vc   = c.vehicleCategory || "Sedan";
  const ac   = c.addressChange || "no change";
  const aph  = c.ageOfHolder || "31 to 35";

  const bpR  = EDA.bp[bp]  ?? 0.06;
  const ftR  = EDA.ft[fa]  ?? 0.06;
  const ptR  = EDA.pt[`${vc} - ${bp}`] ?? 0.06;
  const vcR  = EDA.vc[vc]  ?? 0.06;
  const acR  = EDA.ac[ac]  ?? 0.058;
  const aphR = EDA.aph[aph] ?? 0.06;
  const comp = bpR*0.25 + ftR*0.22 + ptR*0.18 + vcR*0.12 + acR*0.10 + aphR*0.08;

  const isFH = fa === "Policy Holder" ? 1 : 0;
  const isL  = bp.includes("Liability") ? 1 : 0;
  const isAP = bp.includes("All Perils") ? 1 : 0;
  const isTP = 1 - isFH;
  const noP  = (auth === "None" || auth === "Other") ? 1 : 0;
  const noW  = wit === 0 ? 1 : 0;
  const yng  = ["21 to 25","18 to 20","16 to 17"].includes(aph) ? 1 : 0;
  const mvd  = ["under 6 months","2 to 3 years"].includes(ac) ? 1 : 0;
  const sN   = isL + isTP + (wit > 0 ? 1 : 0) + (auth === "Police" ? 1 : 0);
  const rN   = isFH + isAP + mvd + noP + noW + yng;

  const raw =
    comp*0.09964 + sN*-0.05565 + (rN-sN)*0.03704 + ptR*0.02915 +
    bpR*0.025 + ftR*0.022 + isFH*0.050 + isL*-0.045 + isTP*-0.040 +
    (isL*isTP)*-0.080 + noP*0.030 + noW*0.025 + isAP*0.035 + yng*0.030 +
    mvd*0.040 + (isFH*noP)*0.025 + (isAP*noP*noW)*0.020 + (mvd*isFH)*0.015 +
    (prev >= 3 ? 0.020 : 0) + inj*0.040 + (inj > 0.60 ? 0.025 : 0) +
    (amt > 200000 ? 0.015 : 0) + (hr >= 22 || hr <= 4 ? 0.018 : 0) +
    (mo < 6 ? 0.020 : 0);

  return Math.min(0.99, Math.max(0.01, 1 / (1 + Math.exp(-(raw - 0.10) * 12))));
}

const riskColor  = s => s >= 0.70 ? "#dc2626" : s >= 0.35 ? "#d97706" : "#16a34a";
const riskBg     = s => s >= 0.70 ? "#fef2f2" : s >= 0.35 ? "#fffbeb" : "#f0fdf4";
const riskBorder = s => s >= 0.70 ? "#fecaca" : s >= 0.35 ? "#fde68a" : "#bbf7d0";
const riskLabel  = s => s >= 0.70 ? "HIGH" : s >= 0.35 ? "MEDIUM" : "LOW";
const riskAction = s => s >= 0.70 ? "INVESTIGATE" : s >= 0.35 ? "MANUAL REVIEW" : "APPROVE";

function getRiskFactors(c) {
  const inj = Number(c.injuryRatio)||0, hr = Number(c.incidentHour)||0;
  const mo = Number(c.policyMonths)||0, prev = Number(c.prevClaims)||0;
  const amt = Number(c.claimAmount)||0, auth = c.authoritiesContacted||"Police";
  const f = [];
  if (inj > 0.60) f.push({ label:"High injury ratio", detail:`${(inj*100).toFixed(0)}% of claim`, level:"high" });
  if (!Number(c.witnesses||0)) f.push({ label:"No witnesses", detail:"Cannot verify independently", level:"high" });
  if (auth==="None"||auth==="Other") f.push({ label:"No authorities contacted", detail:"No official record", level:"high" });
  if (mo < 12) f.push({ label:"New policy", detail:`${mo} months old`, level:"high" });
  if (prev >= 3) f.push({ label:"Multiple prior claims", detail:`${prev} prior claims`, level:"high" });
  if (hr >= 22 || hr <= 4) f.push({ label:"Late-night incident", detail:`${hr}:00 hrs`, level:"medium" });
  if (amt > 200000) f.push({ label:"High claim amount", detail:`Rs ${Number(amt).toLocaleString("en-IN")}`, level:"medium" });
  if (["under 6 months","2 to 3 years"].includes(c.addressChange)) f.push({ label:"Recent address change", detail:c.addressChange, level:"medium" });
  if (c.basePolicy==="All Perils") f.push({ label:"All Perils policy", detail:"10.2% category fraud rate", level:"medium" });
  if (["21 to 25","18 to 20","16 to 17"].includes(c.ageOfHolder)) f.push({ label:"Young policyholder", detail:c.ageOfHolder, level:"medium" });
  if (auth==="Police" && Number(c.witnesses||0) > 0) f.push({ label:"Police report + witnesses", detail:"Verified incident", level:"safe" });
  if (c.fault==="Third Party") f.push({ label:"Third-party fault", detail:"0.9% category fraud rate", level:"safe" });
  if (c.basePolicy==="Liability") f.push({ label:"Liability policy", detail:"0.7% category fraud rate", level:"safe" });
  if (Number(c.policyMonths||0) > 48) f.push({ label:"Long-standing policy", detail:`${c.policyMonths} months`, level:"safe" });
  return f;
}

// ── Sample claims ─────────────────────────────────────────────────────
const SAMPLES = [
  { id:"CLM-2026-4471", customer:"Ramesh Verma", age:41, basePolicy:"All Perils", fault:"Policy Holder", vehicleCategory:"Sedan", addressChange:"2 to 3 years", ageOfHolder:"41 to 50", policyMonths:5, claimAmount:280000, incidentHour:23, witnesses:0, prevClaims:4, repairShop:"Speedy Auto Works", incidentType:"Multi-vehicle Collision", authoritiesContacted:"None", injuryRatio:0.71, claimText:"I was driving home late at night when another vehicle hit my car from the side at the junction near MG Road. The impact was severe causing significant shoulder and back injury. My vehicle sustained heavy damage. I was hospitalised for 3 days.", history:"Claim 8 months ago: collision Rs 1.4L at Speedy Auto Works. Claim 14 months ago: parked car Rs 0.9L same shop." },
  { id:"CLM-2026-3892", customer:"Priya Nandakumar", age:34, basePolicy:"Collision", fault:"Policy Holder", vehicleCategory:"Sedan", addressChange:"no change", ageOfHolder:"31 to 35", policyMonths:72, claimAmount:42000, incidentHour:11, witnesses:2, prevClaims:0, repairShop:"City Motors Authorized", incidentType:"Single Vehicle", authoritiesContacted:"Police", injuryRatio:0.08, claimText:"I accidentally reversed my car into a pillar in the parking lot of my office building. Two colleagues witnessed the incident. I filed a police report immediately. Rear bumper and boot lid are damaged.", history:"No prior claims. Customer for 6 years. Premiums paid on time." },
  { id:"CLM-2026-5103", customer:"Arjun Menon", age:29, basePolicy:"All Perils", fault:"Policy Holder", vehicleCategory:"Sedan", addressChange:"under 6 months", ageOfHolder:"26 to 30", policyMonths:2, claimAmount:390000, incidentHour:2, witnesses:0, prevClaims:6, repairShop:"Speedy Auto Works", incidentType:"Multi-vehicle Collision", authoritiesContacted:"Ambulance", injuryRatio:0.82, claimText:"A speeding truck hit my vehicle at 2am returning from night shift. The truck fled. I sustained serious chest and neck injuries and was rushed to hospital. My car is completely totalled.", history:"6 claims in 18 months across 2 policies. Policy activated 2 months ago. Previous insurer declined renewal." },
  { id:"CLM-2026-6621", customer:"Kavitha Subramaniam", age:52, basePolicy:"Liability", fault:"Third Party", vehicleCategory:"Sedan", addressChange:"4 to 8 years", ageOfHolder:"51 to 65", policyMonths:96, claimAmount:31000, incidentHour:14, witnesses:3, prevClaims:1, repairShop:"Maruti Authorised Service", incidentType:"Single Vehicle", authoritiesContacted:"Police", injuryRatio:0.12, claimText:"A delivery truck reversed into my parked car outside my office. Three colleagues witnessed it. The at-fault driver's details were recorded by police. Front bumper and bonnet are dented.", history:"One claim 4 years ago, minor damage Rs 18,000. No issues." },
];

const BLANK = { customer:"", age:"", basePolicy:"Collision", fault:"Policy Holder", vehicleCategory:"Sedan", addressChange:"no change", ageOfHolder:"31 to 35", policyMonths:"", claimAmount:"", incidentHour:"", witnesses:"", prevClaims:"", repairShop:"", incidentType:"Single Vehicle", authoritiesContacted:"Police", injuryRatio:"", claimText:"", history:"" };

const AGENTS = [
  { id:"doc",    name:"Document Agent",         color:"#2563eb", bg:"#eff6ff", desc:"Extracts & structures claim fields" },
  { id:"claim",  name:"Claim Analysis",         color:"#7c3aed", bg:"#f5f3ff", desc:"Detects narrative anomalies" },
  { id:"fraud",  name:"Fraud Detection",        color:"#d97706", bg:"#fffbeb", desc:"Scores against trained model" },
  { id:"reason", name:"Investigator Reasoning", color:"#16a34a", bg:"#f0fdf4", desc:"Synthesises evidence" },
  { id:"report", name:"Report Generator",       color:"#dc2626", bg:"#fef2f2", desc:"Formal investigation report" },
];

const RAG_DB = [
  { id:"HIST-001", sim:0.91, verdict:"FRAUD CONFIRMED", vcolor:"#dc2626", desc:"Same-location repeat injury claim filed 8 months after prior identical claim from same customer." },
  { id:"HIST-002", sim:0.87, verdict:"FRAUD CONFIRMED", vcolor:"#dc2626", desc:"Multi-vehicle collision at 11PM, no witnesses, 3-month-old policy, claim 3x vehicle value." },
  { id:"HIST-003", sim:0.79, verdict:"FRAUD RING",      vcolor:"#d97706", desc:"Repair shop cited in 14 separate fraud cases across 3 states in a 12-month period." },
  { id:"HIST-004", sim:0.83, verdict:"SUSPICIOUS",      vcolor:"#d97706", desc:"Injury claim ratio at 78% vs 22% industry average. No hospital admission records found." },
];

// ── Groq API ──────────────────────────────────────────────────────────
async function callGroq(system, user, apiKey, maxTokens = 900) {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: "llama3-70b-8192",
      max_tokens: maxTokens,
      temperature: 0.25,
      messages: [
        { role: "system", content: system },
        { role: "user",   content: user },
      ],
    }),
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.error?.message || `HTTP ${res.status}`); }
  const d = await res.json();
  return d.choices?.[0]?.message?.content || "No response.";
}

// ── Document extraction via Groq ──────────────────────────────────────
async function extractFromDoc(text, apiKey) {
  const system = `You are an insurance document extraction AI. Extract claim data from text and return ONLY valid JSON with keys:
customer, age, basePolicy, fault, vehicleCategory, addressChange, ageOfHolder, policyMonths,
claimAmount, incidentHour, witnesses, prevClaims, repairShop, incidentType, authoritiesContacted, injuryRatio, claimText, history.
Rules: basePolicy ∈ ["Liability","Collision","All Perils"]; fault ∈ ["Policy Holder","Third Party"]; vehicleCategory ∈ ["Sedan","Sport","Utility"]; authoritiesContacted ∈ ["Police","Fire","Ambulance","None","Other"]; injuryRatio decimal 0-1; incidentHour 0-23; null for unknown fields. Return ONLY the JSON object with no extra text.`;
  const out = await callGroq(system, `Extract from:\n\n${text}`, apiKey, 700);
  const clean = out.replace(/```json|```/g, "").trim();
  return JSON.parse(clean);
}

// ── 5-agent pipeline ──────────────────────────────────────────────────
async function runPipeline(claim, apiKey, onStart, onDone) {
  const s = mlScore(claim), lv = riskLabel(s), pct = `${(s*100).toFixed(1)}%`;
  const ctx = [
    `CLAIM: ${claim.id||"CLM-NEW"} | CUSTOMER: ${claim.customer}, Age ${claim.age}`,
    `POLICY: ${claim.basePolicy} · ${claim.vehicleCategory} · Fault: ${claim.fault}`,
    `POLICY AGE: ${claim.policyMonths}mo | ADDRESS CHANGE: ${claim.addressChange} | HOLDER AGE: ${claim.ageOfHolder}`,
    `CLAIM AMOUNT: Rs ${Number(claim.claimAmount).toLocaleString("en-IN")}`,
    `INCIDENT: ${claim.incidentType} at ${claim.incidentHour}:00 | WITNESSES: ${claim.witnesses} | AUTH: ${claim.authoritiesContacted}`,
    `PRIOR CLAIMS: ${claim.prevClaims} | SHOP: ${claim.repairShop} | INJURY RATIO: ${(Number(claim.injuryRatio)*100).toFixed(0)}%`,
    `ML FRAUD SCORE: ${pct} (${lv}) | NARRATIVE: ${claim.claimText}`,
    `HISTORY: ${claim.history || "None"}`,
  ].join("\n");
  const rag = `SIMILAR FRAUD CASES:\n${RAG_DB.map(r=>`- [${r.id} ${(r.sim*100).toFixed(0)}%] ${r.desc} → ${r.verdict}`).join("\n")}`;
  const plain = "Plain text only. No markdown, no asterisks, no bullet points.";

  onStart("doc");
  onDone("doc", await callGroq(
    `You are a document processing agent for insurance fraud detection. ${plain}`,
    `${ctx}\n\nExtract and present: 1) All structured claim fields 2) Missing data 3) Data quality observations 4) Event timeline. ${plain}`, apiKey));

  onStart("claim");
  onDone("claim", await callGroq(
    `You are a claim narrative analyst specialising in insurance fraud. ${plain}`,
    `${ctx}\n\n${rag}\n\nAnalyse: 1) Narrative inconsistencies 2) Suspicious patterns vs fraud cases 3) Red flags 4) Credibility score 0-100 with justification. ${plain}`, apiKey));

  onStart("fraud");
  onDone("fraud", await callGroq(
    `You are a fraud detection AI using rule-based and statistical analysis. ${plain}`,
    `${ctx}\n\nRisk signals:\n- No witnesses: ${Number(claim.witnesses)===0?"YES HIGH":"No"}\n- Late night: ${(Number(claim.incidentHour)>=22||Number(claim.incidentHour)<=4)?"YES HIGH":"No"}\n- New policy <12mo: ${Number(claim.policyMonths)<12?"YES HIGH":"No"}\n- Prior claims >=3: ${Number(claim.prevClaims)>=3?"YES HIGH":"No"}\n- Injury ratio >50%: ${Number(claim.injuryRatio)>0.5?"YES HIGH":"No"}\n- No authorities: ${claim.authoritiesContacted==="None"?"YES HIGH":"No"}\n\nML ensemble: ${pct}\n\nReport: 1) Each triggered signal 2) ML verdict 3) Anomaly findings 4) Final probability. ${plain}`, apiKey));

  onStart("reason");
  onDone("reason", await callGroq(
    `You are a senior insurance fraud investigator with 20 years experience. Professional tone. ${plain}`,
    `${ctx}\n\n1) Does this claim raise serious fraud concerns? 2) Evidence for and against 3) Immediate investigation steps 4) Key interview questions 5) Final verdict with confidence. ${plain}`, apiKey));

  onStart("report");
  const act = s>=0.70?"IMMEDIATE INVESTIGATION REQUIRED":s>=0.35?"MANUAL REVIEW REQUIRED":"APPROVE CLAIM";
  onDone("report", await callGroq(
    `You are a fraud investigation report generator. Formal structured reports. ${plain}`,
    `${ctx}\n\nGenerate formal report using EXACTLY these headings:\n\nEXECUTIVE SUMMARY:\nFRAUD INDICATORS FOUND:\nEVIDENCE ASSESSMENT:\nINVESTIGATION STEPS:\nFINAL DISPOSITION:\n\nFraud risk: ${pct} — ${lv} — ${act}\nUnder 300 words. ${plain}`, apiKey, 800));
}

// ═════════════════════════════════════════════════════════════════════
export default function FraudGuard() {
  const [apiKey, setApiKey]     = useState("");
  const [keyReady, setKeyReady] = useState(false);
  const [mode, setMode]         = useState("sample");
  const [selIdx, setSelIdx]     = useState(0);
  const [form, setForm]         = useState(BLANK);
  const [formErr, setFormErr]   = useState({});
  const [tab, setTab]           = useState("pipeline");
  const [running, setRunning]   = useState(false);
  const [activeA, setActiveA]   = useState(null);
  const [outputs, setOutputs]   = useState({});
  const [done, setDone]         = useState([]);
  const [openA, setOpenA]       = useState(null);
  const [error, setError]       = useState(null);
  const [elapsed, setElapsed]   = useState(0);
  const [audit, setAudit]       = useState([]);
  // Upload
  const [docText, setDocText]   = useState("");
  const [docName, setDocName]   = useState("");
  const [extracting, setExtracting] = useState(false);
  const [extracted, setExtracted]   = useState([]);
  const [extractErr, setExtractErr] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef(null);
  const timerRef = useRef(null);
  const startRef = useRef(null);

  const claim = mode === "sample" ? SAMPLES[selIdx] : { ...BLANK, ...form, id: mode==="upload"?"CLM-DOC":"CLM-NEW" };
  const s  = mlScore(claim);
  const rc = riskColor(s), rb = riskBg(s), rbd = riskBorder(s);
  const rl = riskLabel(s), ra = riskAction(s);
  const factors = getRiskFactors(claim);
  const doneN = done.length, totalN = AGENTS.length;

  useEffect(() => {
    if (running) { startRef.current = Date.now(); timerRef.current = setInterval(() => setElapsed(Date.now()-startRef.current), 100); }
    else clearInterval(timerRef.current);
    return () => clearInterval(timerRef.current);
  }, [running]);

  const reset = () => { setOutputs({}); setDone([]); setOpenA(null); setError(null); setTab("pipeline"); setElapsed(0); };

  const handleFile = (file) => {
    if (!file) return;
    setExtractErr(null); setExtracted([]);
    const reader = new FileReader();
    reader.onload = e => { setDocText(e.target.result); setDocName(file.name); };
    if (file.type === "application/pdf") {
      setExtractErr("PDF detected: Copy-paste the text content from the PDF into the text area below, then click Extract.");
      setDocName(file.name); setDocText("");
    } else {
      reader.readAsText(file);
    }
  };

  const handleExtract = async () => {
    if (!docText.trim()) return;
    setExtracting(true); setExtractErr(null);
    try {
      const data = await extractFromDoc(docText, apiKey);
      const merged = { ...BLANK };
      const fields = [];
      Object.entries(data).forEach(([k,v]) => {
        if (v !== null && v !== undefined && String(v).trim() !== "") {
          merged[k] = String(v); fields.push(k);
        }
      });
      setForm(merged); setExtracted(fields); reset();
    } catch (e) { setExtractErr("Extraction failed: " + e.message); }
    setExtracting(false);
  };

  const validate = () => {
    const e = {};
    if (!form.customer?.trim()) e.customer="Required";
    if (!form.age) e.age="Required";
    if (!form.policyMonths) e.policyMonths="Required";
    if (!form.claimAmount) e.claimAmount="Required";
    if (form.incidentHour==="") e.incidentHour="Required";
    if (form.witnesses==="") e.witnesses="Required";
    if (form.prevClaims==="") e.prevClaims="Required";
    if (!form.repairShop?.trim()) e.repairShop="Required";
    if (!form.injuryRatio) e.injuryRatio="Required";
    if (!form.claimText?.trim()) e.claimText="Required";
    setFormErr(e); return Object.keys(e).length === 0;
  };

  const runAll = async () => {
    if ((mode==="new"||mode==="upload") && !validate()) return;
    setRunning(true); reset();
    try {
      await runPipeline(claim, apiKey, id => setActiveA(id), (id, out) => {
        setOutputs(p => ({...p,[id]:out})); setDone(p => [...p,id]); setActiveA(null);
      });
      const fs = mlScore(claim);
      setAudit(p => [{ id:claim.id||"CLM-NEW", customer:claim.customer||"—", score:fs, verdict:riskLabel(fs), time:new Date().toLocaleTimeString(), amount:claim.claimAmount, source:mode }, ...p.slice(0,19)]);
    } catch (e) { setError(e.message); }
    setActiveA(null); setRunning(false);
  };

  const setF = k => e => { setForm(p=>({...p,[k]:e.target.value})); if(formErr[k]) setFormErr(p=>({...p,[k]:null})); };
  const isExt = k => extracted.includes(k);

  // ── CSS ────────────────────────────────────────────────────────────
  const CSS = `
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#f8f9fb;--s1:#ffffff;--s2:#f1f3f7;--bd:#e2e5ec;--bd2:#d0d5e0;
      --tx:#0f1117;--mt:#6b7280;--sub:#9ca3af;
      --bl:#2563eb;--rd:#dc2626;--am:#d97706;--gn:#16a34a;
      --bl-lt:#eff6ff;--rd-lt:#fef2f2;--am-lt:#fffbeb;--gn-lt:#f0fdf4;
      --shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.05);
      --shadow-md:0 4px 12px rgba(0,0,0,.1),0 2px 4px rgba(0,0,0,.06);
      --mo:'IBM Plex Mono',monospace;--sa:'Plus Jakarta Sans',sans-serif;--di:'Syne',sans-serif
    }
    body{background:var(--bg);color:var(--tx);font-family:var(--sa)}
    ::-webkit-scrollbar{width:4px;height:4px}
    ::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:2px}
    @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
    @keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
    @keyframes scan{0%{transform:translateX(-100%)}100%{transform:translateX(600%)}}
    @keyframes spin{to{transform:rotate(360deg)}}
    @keyframes pulse{0%,100%{opacity:.7}50%{opacity:1}}
    input,select,textarea{
      font-family:var(--mo);font-size:12px;background:var(--s1);border:1.5px solid var(--bd);
      color:var(--tx);border-radius:6px;padding:8px 10px;width:100%;outline:none;
      transition:border-color .15s,box-shadow .15s;font-weight:500
    }
    input:focus,select:focus,textarea:focus{border-color:var(--bl);box-shadow:0 0 0 3px rgba(37,99,235,.1)}
    select{cursor:pointer}
    .ext-field{border-color:#16a34a !important;background:#f0fdf4 !important}
    .btn-primary{
      font-family:var(--di);font-size:13px;font-weight:700;letter-spacing:.03em;
      background:var(--bl);color:#fff;border:none;border-radius:7px;padding:11px;
      cursor:pointer;transition:all .15s;width:100%;box-shadow:var(--shadow)
    }
    .btn-primary:hover:not(:disabled){background:#1d4ed8;box-shadow:var(--shadow-md)}
    .btn-primary:disabled{background:var(--bd2);color:var(--sub);cursor:not-allowed;box-shadow:none}
    .btn-sm{font-family:var(--di);font-size:12px;font-weight:700;background:var(--bl);color:#fff;border:none;border-radius:6px;padding:8px 14px;cursor:pointer;transition:all .15s}
    .btn-sm:hover:not(:disabled){background:#1d4ed8}
    .btn-sm:disabled{background:var(--bd2);color:var(--sub);cursor:not-allowed}
    .btn-ghost{font-family:var(--di);font-size:11px;font-weight:700;background:transparent;border:1.5px solid var(--bd);color:var(--mt);border-radius:6px;padding:6px 12px;cursor:pointer;transition:all .15s}
    .btn-ghost:hover{border-color:var(--bl);color:var(--bl)}
    .mbtn{font-family:var(--di);font-size:12px;font-weight:700;background:transparent;border:1.5px solid var(--bd);color:var(--mt);border-radius:6px;padding:8px 6px;cursor:pointer;transition:all .15s;text-align:center}
    .mbtn.on{background:var(--bl);border-color:var(--bl);color:#fff;box-shadow:var(--shadow)}
    .tbtn{font-family:var(--di);font-size:11px;font-weight:800;letter-spacing:.08em;background:none;border:none;border-bottom:3px solid transparent;padding:13px 16px;color:var(--sub);cursor:pointer;transition:all .15s;text-transform:uppercase}
    .tbtn.on{color:var(--bl);border-bottom-color:var(--bl)}
    .tbtn:hover:not(.on){color:var(--mt)}
    .hr{transition:background .08s;cursor:pointer;border-radius:8px}
    .hr:hover{background:var(--s2)}
    .acard{border-radius:8px;border:1.5px solid var(--bd);padding:14px 12px;cursor:pointer;transition:all .2s;position:relative;overflow:hidden;background:var(--s1)}
    .acard:hover{border-color:var(--bd2);box-shadow:var(--shadow)}
    .drop-zone{border:2px dashed var(--bd2);border-radius:8px;transition:all .15s;cursor:pointer;background:var(--s1)}
    .drop-zone.over,.drop-zone:hover{border-color:var(--bl);background:var(--bl-lt)}
  `;

  const FormFields = () => (
    <div style={{flex:1,overflowY:"auto",padding:"12px 14px"}}>
      {mode==="upload" && extracted.length>0 && (
        <div style={{padding:"8px 12px",background:riskBg(0),border:`1.5px solid ${riskBorder(0)}`,borderRadius:6,marginBottom:12,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <span style={{fontSize:11,fontFamily:"var(--di)",fontWeight:800,color:"var(--gn)"}}>{extracted.length} FIELDS EXTRACTED</span>
          <button className="btn-ghost" onClick={()=>{setDocText("");setExtracted([]);setDocName("");setForm(BLANK);reset();}} style={{padding:"2px 8px",fontSize:10}}>CLEAR</button>
        </div>
      )}
      <LB mb={14}>Claim Details</LB>
      <FD label="Customer Name" err={formErr.customer} ext={isExt("customer")}>
        <input value={form.customer} onChange={setF("customer")} placeholder="Full name" className={isExt("customer")?"ext-field":""} style={{borderColor:formErr.customer?"var(--rd)":undefined}}/>
      </FD>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
        <FD label="Age" err={formErr.age} ext={isExt("age")}>
          <input type="number" value={form.age} onChange={setF("age")} placeholder="35" className={isExt("age")?"ext-field":""}/>
        </FD>
        <FD label="Policy Age (months)" err={formErr.policyMonths} ext={isExt("policyMonths")}>
          <input type="number" value={form.policyMonths} onChange={setF("policyMonths")} placeholder="24" className={isExt("policyMonths")?"ext-field":""}/>
        </FD>
      </div>
      <FD label="Claim Amount (Rs)" err={formErr.claimAmount} ext={isExt("claimAmount")}>
        <input type="number" value={form.claimAmount} onChange={setF("claimAmount")} placeholder="150000" className={isExt("claimAmount")?"ext-field":""} style={{borderColor:formErr.claimAmount?"var(--rd)":undefined}}/>
      </FD>
      {[
        {label:"Base Policy",     key:"basePolicy",           opts:["Liability","Collision","All Perils"]},
        {label:"Fault",           key:"fault",                opts:["Policy Holder","Third Party"]},
        {label:"Vehicle Category",key:"vehicleCategory",      opts:["Sedan","Sport","Utility"]},
        {label:"Address Change",  key:"addressChange",        opts:["no change","under 6 months","1 year","2 to 3 years","4 to 8 years"]},
        {label:"Holder Age Group",key:"ageOfHolder",          opts:["16 to 17","18 to 20","21 to 25","26 to 30","31 to 35","36 to 40","41 to 50","51 to 65","over 65"]},
        {label:"Incident Type",   key:"incidentType",         opts:["Single Vehicle","Multi-vehicle Collision","Parked Car Damage","Theft","Fire","Natural Calamity","Other"]},
        {label:"Authorities",     key:"authoritiesContacted", opts:["Police","Fire","Ambulance","None","Other"]},
      ].map(({label,key,opts}) => (
        <FD key={key} label={label} ext={isExt(key)}>
          <select value={form[key]} onChange={setF(key)} className={isExt(key)?"ext-field":""}>{opts.map(o=><option key={o}>{o}</option>)}</select>
        </FD>
      ))}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
        <FD label="Incident Hour (0-23)" err={formErr.incidentHour} ext={isExt("incidentHour")}>
          <input type="number" value={form.incidentHour} onChange={setF("incidentHour")} placeholder="14" min="0" max="23" className={isExt("incidentHour")?"ext-field":""}/>
        </FD>
        <FD label="Witnesses" err={formErr.witnesses} ext={isExt("witnesses")}>
          <input type="number" value={form.witnesses} onChange={setF("witnesses")} placeholder="0" min="0" className={isExt("witnesses")?"ext-field":""}/>
        </FD>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
        <FD label="Prior Claims" err={formErr.prevClaims} ext={isExt("prevClaims")}>
          <input type="number" value={form.prevClaims} onChange={setF("prevClaims")} placeholder="0" min="0" className={isExt("prevClaims")?"ext-field":""}/>
        </FD>
        <FD label="Injury Ratio (0.0–1.0)" err={formErr.injuryRatio} ext={isExt("injuryRatio")}>
          <input type="number" value={form.injuryRatio} onChange={setF("injuryRatio")} placeholder="0.25" min="0" max="1" step="0.01" className={isExt("injuryRatio")?"ext-field":""}/>
        </FD>
      </div>
      <FD label="Repair Shop" err={formErr.repairShop} ext={isExt("repairShop")}>
        <input value={form.repairShop} onChange={setF("repairShop")} placeholder="Authorised service centre" className={isExt("repairShop")?"ext-field":""} style={{borderColor:formErr.repairShop?"var(--rd)":undefined}}/>
      </FD>
      <FD label="Claim Narrative" err={formErr.claimText} ext={isExt("claimText")}>
        <textarea value={form.claimText} onChange={setF("claimText")} rows={4} placeholder="Customer's account of what happened..." className={isExt("claimText")?"ext-field":""} style={{borderColor:formErr.claimText?"var(--rd)":undefined,resize:"vertical",lineHeight:1.6}}/>
      </FD>
      <FD label="Customer History" ext={isExt("history")}>
        <textarea value={form.history} onChange={setF("history")} rows={2} placeholder="Prior claims or notes..." className={isExt("history")?"ext-field":""} style={{resize:"vertical",lineHeight:1.6}}/>
      </FD>
      {form.claimAmount && form.incidentHour !== "" && (
        <div style={{padding:12,background:rb,border:`1.5px solid ${rbd}`,borderRadius:7,marginBottom:4,animation:"fadeIn .3s ease"}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
            <span style={{fontSize:10,fontFamily:"var(--di)",fontWeight:800,color:rc,letterSpacing:".12em"}}>LIVE ESTIMATE</span>
            <span style={{fontSize:14,fontFamily:"var(--di)",fontWeight:800,color:rc}}>{(s*100).toFixed(0)}% {rl}</span>
          </div>
          <div style={{height:4,background:"var(--bd)",borderRadius:2,overflow:"hidden"}}>
            <div style={{width:`${s*100}%`,height:"100%",background:rc,borderRadius:2,transition:"width .3s"}}/>
          </div>
        </div>
      )}
    </div>
  );

  // ── API key gate ────────────────────────────────────────────────────
  if (!keyReady) return (
    <div style={{minHeight:"100vh",background:"var(--bg)",display:"flex",alignItems:"center",justifyContent:"center",fontFamily:"var(--sa)"}}>
      <style>{CSS}</style>
      <div style={{width:440,background:"var(--s1)",borderRadius:14,border:"1.5px solid var(--bd)",boxShadow:"0 20px 60px rgba(0,0,0,.12)",padding:36}}>
        <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:24}}>
          <div style={{width:40,height:40,background:"var(--bl)",borderRadius:10,display:"flex",alignItems:"center",justifyContent:"center",boxShadow:"0 4px 12px rgba(37,99,235,.35)"}}>
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M9 1.5L15 5V13L9 16.5L3 13V5L9 1.5Z" stroke="white" strokeWidth="1.5" fill="none"/><circle cx="9" cy="9" r="2" fill="white"/></svg>
          </div>
          <div>
            <div style={{fontSize:22,fontFamily:"var(--di)",fontWeight:800,color:"var(--tx)"}}>FraudGuard</div>
            <div style={{fontSize:11,fontFamily:"var(--mo)",fontWeight:600,color:"var(--sub)",letterSpacing:".1em"}}>INSURANCE FRAUD DETECTION</div>
          </div>
        </div>
        <div style={{padding:"12px 14px",background:"var(--bl-lt)",borderRadius:7,border:"1.5px solid #bfdbfe",marginBottom:22}}>
          <div style={{fontSize:10,fontFamily:"var(--di)",fontWeight:800,color:"var(--bl)",letterSpacing:".12em",marginBottom:4}}>GROQ API — LLAMA 3 70B</div>
          <div style={{fontSize:12,color:"#1e40af",lineHeight:1.6}}>Free API key at <strong>console.groq.com</strong> — no credit card. Runs Llama 3 70B instantly.</div>
        </div>
        <LB mb={8}>Groq API Key</LB>
        <input type="password" placeholder="gsk_..." value={apiKey} onChange={e=>setApiKey(e.target.value)} onKeyDown={e=>e.key==="Enter"&&apiKey.trim()&&setKeyReady(true)} style={{marginBottom:14,fontSize:13}}/>
        <button className="btn-primary" onClick={()=>apiKey.trim()&&setKeyReady(true)} disabled={!apiKey.trim()}>Enter FraudGuard</button>
        <div style={{marginTop:20,display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
          {[["4 Sample Claims","Pre-loaded realistic cases"],["Document Upload","PDF/text extraction via AI"],["5 AI Agents","Full investigation pipeline"],["Audit Log","Session tracking"]].map(([t,d])=>(
            <div key={t} style={{padding:"10px 12px",background:"var(--s2)",borderRadius:6,border:"1.5px solid var(--bd)"}}>
              <div style={{fontSize:11,fontFamily:"var(--di)",fontWeight:800,color:"var(--tx)",marginBottom:2}}>{t}</div>
              <div style={{fontSize:10,color:"var(--mt)"}}>{d}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  // ── Main app ────────────────────────────────────────────────────────
  return (
    <div style={{minHeight:"100vh",background:"var(--bg)",color:"var(--tx)",fontFamily:"var(--sa)"}}>
      <style>{CSS}</style>

      {/* HEADER */}
      <div style={{height:52,background:"var(--s1)",borderBottom:"1.5px solid var(--bd)",display:"flex",alignItems:"center",justifyContent:"space-between",padding:"0 22px",boxShadow:"0 1px 4px rgba(0,0,0,.06)",position:"sticky",top:0,zIndex:50}}>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{width:30,height:30,background:"var(--bl)",borderRadius:7,display:"flex",alignItems:"center",justifyContent:"center",boxShadow:"0 2px 8px rgba(37,99,235,.4)"}}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 1L12 3.5V10.5L7 13L2 10.5V3.5L7 1Z" stroke="white" strokeWidth="1.3" fill="none"/><circle cx="7" cy="7" r="1.8" fill="white"/></svg>
          </div>
          <div>
            <div style={{fontSize:15,fontFamily:"var(--di)",fontWeight:800,color:"var(--tx)",letterSpacing:"-.01em"}}>FraudGuard</div>
            <div style={{fontSize:8,fontFamily:"var(--mo)",fontWeight:600,color:"var(--sub)",letterSpacing:".14em"}}>RF+ET+GB · AUC 0.8381 · LLAMA 3 70B</div>
          </div>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:14}}>
          {doneN===totalN && (
            <div style={{display:"flex",alignItems:"center",gap:7,padding:"5px 12px",borderRadius:6,background:rb,border:`1.5px solid ${rbd}`,animation:"fadeIn .3s ease"}}>
              <div style={{width:7,height:7,borderRadius:"50%",background:rc}}/>
              <span style={{fontSize:11,fontFamily:"var(--di)",fontWeight:800,color:rc,letterSpacing:".06em"}}>{rl} · {(s*100).toFixed(1)}%</span>
            </div>
          )}
          {running && (
            <div style={{display:"flex",alignItems:"center",gap:7}}>
              <div style={{width:12,height:12,border:"2px solid var(--bl)",borderTopColor:"transparent",borderRadius:"50%",animation:"spin .8s linear infinite"}}/>
              <span style={{fontSize:11,fontFamily:"var(--mo)",fontWeight:600,color:"var(--bl)",animation:"blink 1s infinite"}}>{(elapsed/1000).toFixed(1)}s</span>
            </div>
          )}
          <div style={{display:"flex",alignItems:"center",gap:6,padding:"4px 10px",borderRadius:5,background:"var(--gn-lt)",border:"1.5px solid #bbf7d0"}}>
            <div style={{width:6,height:6,borderRadius:"50%",background:"var(--gn)"}}/>
            <span style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:"var(--gn)",letterSpacing:".1em"}}>GROQ · LLAMA 3 70B</span>
          </div>
        </div>
      </div>

      {/* LAYOUT */}
      <div style={{display:"grid",gridTemplateColumns:"306px 1fr",height:"calc(100vh - 52px)"}}>

        {/* LEFT */}
        <div style={{background:"var(--s1)",borderRight:"1.5px solid var(--bd)",display:"flex",flexDirection:"column",overflow:"hidden",boxShadow:"1px 0 4px rgba(0,0,0,.04)"}}>

          {/* Mode tabs */}
          <div style={{padding:"14px 14px 10px",borderBottom:"1.5px solid var(--bd)",background:"var(--s2)"}}>
            <LB mb={8}>Claim Source</LB>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:5}}>
              {[["sample","Samples"],["new","Manual"],["upload","Upload"]].map(([m,l])=>(
                <button key={m} className={`mbtn${mode===m?" on":""}`} onClick={()=>{setMode(m);reset();}}>{l}</button>
              ))}
            </div>
          </div>

          {/* SAMPLE */}
          {mode==="sample" && (<>
            <div style={{padding:"10px 14px",borderBottom:"1.5px solid var(--bd)"}}>
              {SAMPLES.map((c,i) => {
                const cs=mlScore(c),col=riskColor(cs),bg=riskBg(cs),bdr=riskBorder(cs),lbl=riskLabel(cs),active=selIdx===i;
                return (
                  <div key={c.id} className="hr" onClick={()=>{setSelIdx(i);reset();}} style={{padding:"10px 10px",marginBottom:5,background:active?bg:"transparent",border:`1.5px solid ${active?bdr:"transparent"}`,boxShadow:active?"var(--shadow)":"none"}}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:3}}>
                      <span style={{fontSize:12,fontFamily:"var(--di)",fontWeight:800,color:active?col:"var(--tx)"}}>{c.customer}</span>
                      <span style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:col,background:bg,border:`1.5px solid ${bdr}`,padding:"2px 7px",borderRadius:4,letterSpacing:".06em"}}>{lbl}</span>
                    </div>
                    <div style={{fontSize:9,fontFamily:"var(--mo)",fontWeight:600,color:"var(--sub)",marginBottom:2}}>{c.id}</div>
                    <div style={{fontSize:10,color:"var(--mt)",fontWeight:600}}>Rs {(c.claimAmount/1e5).toFixed(1)}L · {c.incidentType}</div>
                    {active && <div style={{marginTop:6,height:3,background:"var(--bd)",borderRadius:2,overflow:"hidden"}}><div style={{width:`${cs*100}%`,height:"100%",background:col,borderRadius:2}}/></div>}
                  </div>
                );
              })}
            </div>
            <div style={{flex:1,overflowY:"auto",padding:"12px 14px"}}>
              <LB mb={10}>Claim Details</LB>
              {[["Policy",`${SAMPLES[selIdx].basePolicy} · ${SAMPLES[selIdx].vehicleCategory}`],["Fault",SAMPLES[selIdx].fault],["Amount",`Rs ${SAMPLES[selIdx].claimAmount.toLocaleString("en-IN")}`],["Policy Age",`${SAMPLES[selIdx].policyMonths} months`],["Incident",SAMPLES[selIdx].incidentType],["Time",`${SAMPLES[selIdx].incidentHour}:00 hrs`],["Witnesses",SAMPLES[selIdx].witnesses],["Prior Claims",SAMPLES[selIdx].prevClaims],["Authorities",SAMPLES[selIdx].authoritiesContacted],["Injury Ratio",`${(SAMPLES[selIdx].injuryRatio*100).toFixed(0)}%`]].map(([k,v])=>(
                <div key={k} style={{display:"flex",justifyContent:"space-between",padding:"6px 0",borderBottom:"1px solid var(--bd)"}}>
                  <span style={{fontSize:11,fontWeight:700,color:"var(--mt)"}}>{k}</span>
                  <span style={{fontSize:11,fontFamily:"var(--mo)",fontWeight:600,color:"var(--tx)",textAlign:"right",marginLeft:8}}>{v}</span>
                </div>
              ))}
              <div style={{marginTop:12,padding:12,background:rb,border:`1.5px solid ${rbd}`,borderRadius:8}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:7}}>
                  <span style={{fontSize:10,fontFamily:"var(--di)",fontWeight:800,color:rc,letterSpacing:".12em"}}>PRE-SCORE</span>
                  <span style={{fontSize:18,fontFamily:"var(--di)",fontWeight:800,color:rc}}>{(s*100).toFixed(0)}% {rl}</span>
                </div>
                <div style={{height:4,background:"var(--bd)",borderRadius:2,overflow:"hidden"}}>
                  <div style={{width:`${s*100}%`,height:"100%",background:rc,borderRadius:2}}/>
                </div>
              </div>
              <div style={{marginTop:10,padding:12,background:"var(--s2)",borderRadius:7,border:"1.5px solid var(--bd)"}}>
                <LB mb={6}>Narrative</LB>
                <p style={{fontSize:11,color:"var(--mt)",lineHeight:1.65,fontWeight:500}}>{SAMPLES[selIdx].claimText.slice(0,200)}…</p>
              </div>
            </div>
          </>)}

          {/* NEW */}
          {mode==="new" && FormFields()}

          {/* UPLOAD */}
          {mode==="upload" && (<>
            <div style={{padding:"12px 14px",borderBottom:"1.5px solid var(--bd)"}}>
              <LB mb={10}>Upload Claim Document</LB>
              <div
                className={`drop-zone${dragOver?" over":""}`}
                onDragOver={e=>{e.preventDefault();setDragOver(true)}}
                onDragLeave={()=>setDragOver(false)}
                onDrop={e=>{e.preventDefault();setDragOver(false);handleFile(e.dataTransfer.files[0])}}
                onClick={()=>fileRef.current?.click()}
                style={{padding:"20px 14px",textAlign:"center",marginBottom:10}}
              >
                <input ref={fileRef} type="file" accept=".txt,.csv,.pdf,text/*" style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>
                <svg width="28" height="28" viewBox="0 0 28 28" fill="none" style={{margin:"0 auto 8px",display:"block"}}><rect x="4" y="5" width="20" height="18" rx="2" stroke="#9ca3af" strokeWidth="1.4" fill="none"/><path d="M14 9v10M10 13l4-4 4 4" stroke="#9ca3af" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/></svg>
                {docName ? <div style={{fontSize:11,fontFamily:"var(--di)",fontWeight:800,color:"var(--bl)"}}>{docName}</div> : <div style={{fontSize:11,fontFamily:"var(--di)",fontWeight:700,color:"var(--sub)"}}>Drop file or click to browse<br/><span style={{fontWeight:500,fontSize:10}}>TXT · CSV · PDF (paste text)</span></div>}
              </div>
              {extractErr && <div style={{fontSize:10,color:"var(--rd)",fontFamily:"var(--mo)",fontWeight:600,marginBottom:8,padding:"7px 10px",background:"var(--rd-lt)",borderRadius:5,border:"1.5px solid #fecaca"}}>{extractErr}</div>}
              <div style={{marginBottom:8}}>
                <LB mb={5}>Document Text</LB>
                <textarea value={docText} onChange={e=>setDocText(e.target.value)} rows={5} placeholder="Paste document text here, or drop a .txt file above..." style={{resize:"vertical",lineHeight:1.55,fontSize:11}}/>
              </div>
              <button className="btn-sm" onClick={handleExtract} disabled={extracting||!docText.trim()} style={{width:"100%"}}>
                {extracting ? <span style={{display:"flex",alignItems:"center",justifyContent:"center",gap:8}}><span style={{width:10,height:10,border:"2px solid rgba(255,255,255,.3)",borderTopColor:"#fff",borderRadius:"50%",animation:"spin .8s linear infinite",display:"inline-block"}}/> EXTRACTING...</span> : "EXTRACT CLAIM DATA WITH AI"}
              </button>
            </div>
            {FormFields()}
          </>)}

          {/* Run button */}
          <div style={{padding:14,borderTop:"1.5px solid var(--bd)",background:"var(--s2)"}}>
            {running && (
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
                <div style={{display:"flex",gap:4}}>
                  {AGENTS.map(a=>(
                    <div key={a.id} style={{width:7,height:7,borderRadius:"50%",background:done.includes(a.id)?a.color:activeA===a.id?a.color:"var(--bd2)",transition:"background .3s",animation:activeA===a.id?"blink .8s infinite":"none"}}/>
                  ))}
                </div>
                <span style={{fontSize:10,fontFamily:"var(--di)",fontWeight:800,color:"var(--sub)"}}>{doneN}/{totalN} AGENTS</span>
              </div>
            )}
            <button className="btn-primary" onClick={runAll} disabled={running||(mode==="upload"&&!extracted.length&&!form.claimText)}>
              {running?`ANALYSING — ${doneN}/${totalN}`:doneN===totalN?"RE-RUN PIPELINE":"RUN AGENT PIPELINE"}
            </button>
            {mode==="upload"&&!extracted.length&&!form.claimText && <div style={{marginTop:5,fontSize:10,fontFamily:"var(--di)",fontWeight:700,color:"var(--sub)",textAlign:"center"}}>Extract document first</div>}
          </div>
        </div>

        {/* RIGHT */}
        <div style={{display:"flex",flexDirection:"column",overflow:"hidden"}}>
          <div style={{background:"var(--s1)",borderBottom:"1.5px solid var(--bd)",display:"flex",padding:"0 22px",gap:2,boxShadow:"0 1px 4px rgba(0,0,0,.04)"}}>
            {[["pipeline","Agent Pipeline"],["breakdown","Risk Breakdown"],["rag","Model & RAG"],["report","Report"],["audit","Audit Log"]].map(([id,label])=>(
              <button key={id} className={`tbtn${tab===id?" on":""}`} onClick={()=>setTab(id)}>{label}</button>
            ))}
          </div>

          <div style={{flex:1,overflowY:"auto",padding:22}}>

            {/* ── PIPELINE ── */}
            {tab==="pipeline" && (<>
              <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:8,marginBottom:22}}>
                {AGENTS.map((agent,i)=>{
                  const isDone=done.includes(agent.id),isActive=activeA===agent.id,isOpen=openA===agent.id;
                  return (
                    <div key={agent.id} className="acard" onClick={()=>isDone&&setOpenA(isOpen?null:agent.id)} style={{background:isDone?agent.bg:isActive?"#fafbfc":"var(--s1)",borderColor:isDone?agent.color+"60":isActive?agent.color+"80":"var(--bd)",boxShadow:isDone?"var(--shadow)":"none",cursor:isDone?"pointer":"default"}}>
                      {isActive && <div style={{position:"absolute",top:0,left:0,right:0,height:3,background:agent.color,overflow:"hidden"}}><div style={{position:"absolute",width:"35%",height:"100%",background:"rgba(255,255,255,.6)",animation:"scan 1.5s linear infinite"}}/></div>}
                      <div style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:"var(--sub)",letterSpacing:".14em",marginBottom:6}}>AGENT {i+1}</div>
                      <div style={{fontSize:11,fontFamily:"var(--di)",fontWeight:800,color:isDone?agent.color:isActive?agent.color:"var(--sub)",marginBottom:5,lineHeight:1.3}}>{agent.name}</div>
                      <div style={{fontSize:10,color:"var(--mt)",lineHeight:1.5,marginBottom:10,fontWeight:500}}>{agent.desc}</div>
                      <div style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:isDone?agent.color:isActive?agent.color:"var(--bd2)",letterSpacing:".1em",animation:isActive?"blink .8s infinite":"none"}}>
                        {isDone?"COMPLETE":isActive?"RUNNING":"WAITING"}
                      </div>
                    </div>
                  );
                })}
              </div>

              {doneN>0 && (
                <div style={{background:rb,border:`1.5px solid ${rbd}`,borderRadius:10,padding:"18px 22px",marginBottom:20,display:"flex",alignItems:"center",gap:24,boxShadow:"var(--shadow-md)",animation:"fadeIn .4s ease"}}>
                  <div style={{minWidth:90}}>
                    <div style={{fontSize:48,fontFamily:"var(--di)",fontWeight:800,color:rc,lineHeight:1}}>{(s*100).toFixed(0)}%</div>
                    <div style={{fontSize:10,fontFamily:"var(--di)",fontWeight:800,color:rc,letterSpacing:".14em",marginTop:2}}>FRAUD RISK</div>
                  </div>
                  <div style={{flex:1}}>
                    <div style={{height:8,background:"var(--bd)",borderRadius:4,overflow:"hidden",marginBottom:8}}>
                      <div style={{width:`${s*100}%`,height:"100%",background:`linear-gradient(90deg,var(--gn) 0%,var(--am) 50%,var(--rd) 100%)`,borderRadius:4,transition:"width .6s"}}/>
                    </div>
                    <div style={{display:"flex",justifyContent:"space-between"}}>
                      <span style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:"var(--gn)",letterSpacing:".08em"}}>0–34% AUTO APPROVE</span>
                      <span style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:"var(--am)",letterSpacing:".08em"}}>35–69% MANUAL REVIEW</span>
                      <span style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:"var(--rd)",letterSpacing:".08em"}}>70%+ INVESTIGATE</span>
                    </div>
                  </div>
                  <div>
                    <div style={{fontSize:12,fontFamily:"var(--di)",fontWeight:800,color:rc,background:"white",border:`2px solid ${rbd}`,padding:"7px 16px",borderRadius:7,textAlign:"center",marginBottom:5,boxShadow:"var(--shadow)"}}>{rl}</div>
                    <div style={{fontSize:11,fontFamily:"var(--di)",fontWeight:700,color:rc,textAlign:"center"}}>{ra}</div>
                  </div>
                </div>
              )}

              {doneN===0&&!running && (
                <div style={{textAlign:"center",padding:"72px 20px",animation:"fadeIn .4s ease"}}>
                  <div style={{width:56,height:56,borderRadius:12,border:"2px solid var(--bd)",background:"var(--s2)",margin:"0 auto 18px",display:"flex",alignItems:"center",justifyContent:"center"}}>
                    <svg width="22" height="22" viewBox="0 0 22 22" fill="none"><path d="M11 2L18 6V16L11 20L4 16V6L11 2Z" stroke="#d1d5db" strokeWidth="1.5" fill="none"/><circle cx="11" cy="11" r="2.5" fill="#d1d5db"/></svg>
                  </div>
                  <div style={{fontSize:15,fontFamily:"var(--di)",fontWeight:800,color:"var(--sub)",marginBottom:7}}>{mode==="upload"?"Upload a document and extract data, then run":"Select or fill a claim and run the pipeline"}</div>
                  <div style={{fontSize:12,color:"var(--mt)",fontWeight:500}}>5 Llama 3 70B agents analyse end-to-end</div>
                </div>
              )}

              <div style={{display:"flex",flexDirection:"column",gap:6}}>
                {AGENTS.map(agent=>{
                  const out=outputs[agent.id]; if(!out) return null;
                  const isOpen=openA===agent.id;
                  return (
                    <div key={agent.id} style={{borderRadius:9,overflow:"hidden",border:`1.5px solid ${isOpen?agent.color+"50":"var(--bd)"}`,boxShadow:isOpen?"var(--shadow-md)":"var(--shadow)",animation:"fadeIn .3s ease"}}>
                      <div className="hr" onClick={()=>setOpenA(isOpen?null:agent.id)} style={{padding:"11px 16px",display:"flex",alignItems:"center",gap:10,background:"var(--s1)"}}>
                        <div style={{width:8,height:8,borderRadius:"50%",background:agent.color,flexShrink:0}}/>
                        <div style={{flex:1,minWidth:0}}>
                          <div style={{fontSize:11,fontFamily:"var(--di)",fontWeight:800,color:agent.color,marginBottom:1}}>{agent.name}</div>
                          <div style={{fontSize:10,color:"var(--mt)",overflow:"hidden",whiteSpace:"nowrap",textOverflow:"ellipsis",fontWeight:500}}>{out.slice(0,120)}…</div>
                        </div>
                        <div style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:agent.color,padding:"3px 8px",border:`1.5px solid ${agent.color}30`,background:agent.bg,borderRadius:4,flexShrink:0}}>{isOpen?"HIDE":"VIEW"}</div>
                      </div>
                      {isOpen && (
                        <div style={{padding:"16px 18px",background:"var(--s2)",borderTop:`1.5px solid ${agent.color}20`}}>
                          <pre style={{fontSize:11,fontFamily:"var(--mo)",color:"var(--tx)",whiteSpace:"pre-wrap",lineHeight:1.85,fontWeight:500}}>{out}</pre>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              {error && <div style={{marginTop:12,padding:13,background:"var(--rd-lt)",border:"1.5px solid #fecaca",borderRadius:7,fontSize:11,fontFamily:"var(--mo)",fontWeight:600,color:"var(--rd)"}}>{error}</div>}
            </>)}

            {/* ── BREAKDOWN ── */}
            {tab==="breakdown" && (<>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12,marginBottom:20}}>
                {[["Fraud Probability",`${(s*100).toFixed(1)}%`,rc,rb,rbd,true],["Risk Level",rl,rc,rb,rbd,false],["Action",ra,rc,rb,rbd,false]].map(([label,val,color,bg,bdr,big])=>(
                  <div key={label} style={{padding:18,background:bg,border:`1.5px solid ${bdr}`,borderRadius:10,boxShadow:"var(--shadow)"}}>
                    <LB c={color} mb={6}>{label}</LB>
                    <div style={{fontSize:big?44:20,fontFamily:"var(--di)",fontWeight:800,color,lineHeight:1.1}}>{val}</div>
                  </div>
                ))}
              </div>

              <div style={{background:"var(--s1)",border:"1.5px solid var(--bd)",borderRadius:10,padding:18,marginBottom:16,boxShadow:"var(--shadow)"}}>
                <LB mb={14}>Risk Factor Analysis</LB>
                {factors.length===0 && <div style={{fontSize:12,color:"var(--mt)",fontWeight:500}}>No significant risk factors identified.</div>}
                {["high","medium","safe"].map(level=>{
                  const group=factors.filter(f=>f.level===level); if(!group.length) return null;
                  const color=level==="high"?"var(--rd)":level==="medium"?"var(--am)":"var(--gn)";
                  const bg=level==="high"?"var(--rd-lt)":level==="medium"?"var(--am-lt)":"var(--gn-lt)";
                  const label=level==="high"?"HIGH RISK INDICATORS":level==="medium"?"ELEVATED FACTORS":"MITIGATING FACTORS";
                  return (
                    <div key={level} style={{marginBottom:18}}>
                      <LB c={color} mb={8}>{label}</LB>
                      {group.map((f,i)=>(
                        <div key={i} style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 12px",background:bg,borderRadius:7,marginBottom:5,border:`1.5px solid ${color}25`}}>
                          <div>
                            <div style={{fontSize:12,fontFamily:"var(--di)",fontWeight:800,color:"var(--tx)"}}>{f.label}</div>
                            <div style={{fontSize:10,color:"var(--mt)",marginTop:2,fontWeight:500}}>{f.detail}</div>
                          </div>
                          <div style={{width:10,height:10,borderRadius:"50%",background:color,flexShrink:0,marginLeft:12}}/>
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>

              <div style={{background:"var(--s1)",border:"1.5px solid var(--bd)",borderRadius:10,padding:18,boxShadow:"var(--shadow)"}}>
                <LB mb={14}>Probability Gauge</LB>
                <div style={{position:"relative",height:40,marginBottom:12}}>
                  <div style={{height:10,background:"linear-gradient(90deg,var(--gn),var(--am) 50%,var(--rd))",borderRadius:5,marginTop:15}}/>
                  <div style={{position:"absolute",top:0,left:`${Math.max(0,Math.min(97,s*100))}%`,width:3,height:40,background:"var(--tx)",borderRadius:2,transform:"translateX(-1.5px)",boxShadow:"0 0 8px rgba(0,0,0,.3)"}}/>
                </div>
                <div style={{display:"flex",justifyContent:"space-between",marginBottom:16}}>
                  {["0%","25%","50%","75%","100%"].map(l=><span key={l} style={{fontSize:10,fontFamily:"var(--mo)",fontWeight:600,color:"var(--sub)"}}>{l}</span>)}
                </div>
                <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:10}}>
                  {[["0–34%","Auto Approve","var(--gn)","var(--gn-lt)","#bbf7d0"],["35–69%","Manual Review","var(--am)","var(--am-lt)","#fde68a"],["70–100%","Investigate","var(--rd)","var(--rd-lt)","#fecaca"]].map(([range,action,col,bg,bdr])=>(
                    <div key={range} style={{padding:"11px 12px",background:bg,border:`1.5px solid ${bdr}`,borderRadius:8,textAlign:"center"}}>
                      <div style={{fontSize:13,fontFamily:"var(--di)",fontWeight:800,color:col}}>{range}</div>
                      <div style={{fontSize:10,color:"var(--mt)",marginTop:3,fontWeight:600}}>{action}</div>
                    </div>
                  ))}
                </div>
              </div>
            </>)}

            {/* ── MODEL & RAG ── */}
            {tab==="rag" && (<>
              <div style={{background:"var(--s1)",border:"1.5px solid var(--bd)",borderRadius:10,padding:18,marginBottom:16,boxShadow:"var(--shadow)"}}>
                <LB mb={14}>Trained Ensemble Model · 15,420 Real Claims</LB>
                <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:8,marginBottom:18}}>
                  {[["Accuracy","94.20%","#2563eb","#eff6ff","#bfdbfe"],["ROC-AUC","0.8381","#7c3aed","#f5f3ff","#ddd6fe"],["Precision","87.5%","#16a34a","#f0fdf4","#bbf7d0"],["Recall","71.4%","#d97706","#fffbeb","#fde68a"],["F1 Score","0.786","#dc2626","#fef2f2","#fecaca"]].map(([l,v,c,bg,bdr])=>(
                    <div key={l} style={{background:bg,border:`1.5px solid ${bdr}`,borderRadius:8,padding:"12px 10px",textAlign:"center",boxShadow:"var(--shadow)"}}>
                      <div style={{fontSize:18,fontFamily:"var(--di)",fontWeight:800,color:c}}>{v}</div>
                      <div style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:c,marginTop:3,letterSpacing:".1em"}}>{l}</div>
                    </div>
                  ))}
                </div>
                <LB mb={12}>Feature Importances (Ensemble)</LB>
                {FEAT_IMP.map(({name,v})=>(
                  <div key={name} style={{display:"flex",alignItems:"center",gap:10,marginBottom:7}}>
                    <div style={{width:155,fontSize:10,fontFamily:"var(--mo)",fontWeight:600,color:"var(--mt)"}}>{name}</div>
                    <div style={{flex:1,height:6,background:"var(--s2)",borderRadius:3,border:"1px solid var(--bd)"}}>
                      <div style={{width:`${(v/0.30)*100}%`,height:"100%",background:"var(--bl)",borderRadius:3}}/>
                    </div>
                    <div style={{width:40,fontSize:10,fontFamily:"var(--mo)",fontWeight:700,color:"var(--bl)",textAlign:"right"}}>{v.toFixed(4)}</div>
                  </div>
                ))}
                <div style={{marginTop:16,padding:14,background:"var(--s2)",borderRadius:7,border:"1.5px solid var(--bd)",display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
                  {[["Architecture","RF (55%) + ExtraTrees (15%) + GradientBoost (30%)"],["Training Data","fraud_oracle.csv + carclaims.csv · 15,420 rows"],["CV Score","0.8516 ± 0.012 (5-fold cross-validation)"],["Threshold","0.65 — precision-optimised"]].map(([k,v])=>(
                    <div key={k}>
                      <div style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:"var(--sub)",letterSpacing:".12em",marginBottom:3}}>{k}</div>
                      <div style={{fontSize:11,color:"var(--tx)",fontWeight:600}}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{background:"var(--s1)",border:"1.5px solid var(--bd)",borderRadius:10,padding:18,marginBottom:16,boxShadow:"var(--shadow)"}}>
                <LB mb={14}>EDA Fraud Rates by Category</LB>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14}}>
                  {[{label:"Base Policy",data:EDA.bp},{label:"Fault Type",data:EDA.ft},{label:"Vehicle",data:EDA.vc}].map(({label,data})=>(
                    <div key={label}>
                      <LB mb={10}>{label}</LB>
                      {Object.entries(data).map(([k,v])=>(
                        <div key={k} style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:7}}>
                          <span style={{fontSize:10,color:"var(--mt)",fontWeight:600}}>{k}</span>
                          <div style={{display:"flex",alignItems:"center",gap:6}}>
                            <div style={{width:44,height:5,background:"var(--bd)",borderRadius:2}}>
                              <div style={{width:`${Math.min(100,(v/0.15)*100)}%`,height:"100%",background:v>0.08?"var(--rd)":v>0.04?"var(--am)":"var(--gn)",borderRadius:2}}/>
                            </div>
                            <span style={{fontSize:10,fontFamily:"var(--mo)",fontWeight:700,color:v>0.08?"var(--rd)":v>0.04?"var(--am)":"var(--gn)",minWidth:32,textAlign:"right"}}>{(v*100).toFixed(1)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>

              <div style={{background:"var(--s1)",border:"1.5px solid var(--bd)",borderRadius:10,padding:18,boxShadow:"var(--shadow)"}}>
                <LB mb={6}>Vector Database — Historical Fraud Cases</LB>
                <div style={{fontSize:11,color:"var(--mt)",marginBottom:14,lineHeight:1.65,fontWeight:500}}>Semantically similar past cases retrieved by cosine similarity and injected into agent prompts as RAG context.</div>
                <div style={{background:"var(--bl-lt)",borderRadius:7,padding:12,border:"1.5px solid #bfdbfe",marginBottom:14,fontFamily:"var(--mo)",fontSize:11,color:"var(--bl)",lineHeight:1.9,fontWeight:600}}>
                  embed("{(claim.claimText||"Enter narrative…").slice(0,54)}…")<br/>
                  <span style={{color:"var(--sub)"}}>search(collection="fraud_cases", top_k=4)</span>
                </div>
                {RAG_DB.map(c=>(
                  <div key={c.id} style={{background:"var(--s2)",border:"1.5px solid var(--bd)",borderRadius:8,padding:14,marginBottom:8,boxShadow:"var(--shadow)"}}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
                      <div style={{display:"flex",gap:8,alignItems:"center"}}>
                        <span style={{fontSize:9,fontFamily:"var(--mo)",fontWeight:700,color:"var(--sub)"}}>{c.id}</span>
                        <span style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:c.vcolor,background:`${c.vcolor}15`,border:`1.5px solid ${c.vcolor}30`,padding:"2px 7px",borderRadius:4}}>{c.verdict}</span>
                      </div>
                      <div style={{display:"flex",alignItems:"center",gap:7}}>
                        <div style={{width:56,height:5,background:"var(--bd)",borderRadius:2,overflow:"hidden"}}>
                          <div style={{width:`${c.sim*100}%`,height:"100%",background:"var(--bl)",borderRadius:2}}/>
                        </div>
                        <span style={{fontSize:10,fontFamily:"var(--mo)",fontWeight:700,color:"var(--bl)"}}>{(c.sim*100).toFixed(0)}%</span>
                      </div>
                    </div>
                    <div style={{fontSize:11,color:"var(--mt)",lineHeight:1.6,fontWeight:500}}>{c.desc}</div>
                  </div>
                ))}
              </div>
            </>)}

            {/* ── REPORT ── */}
            {tab==="report" && (
              !outputs.report ? (
                <div style={{textAlign:"center",padding:"72px 20px"}}>
                  <div style={{width:52,height:52,borderRadius:10,border:"2px solid var(--bd)",background:"var(--s2)",margin:"0 auto 16px",display:"flex",alignItems:"center",justifyContent:"center"}}>
                    <svg width="20" height="20" viewBox="0 0 20 20" fill="none"><rect x="4" y="3" width="12" height="14" rx="1.5" stroke="#d1d5db" strokeWidth="1.5" fill="none"/><line x1="7" y1="7" x2="13" y2="7" stroke="#d1d5db" strokeWidth="1.2"/><line x1="7" y1="10" x2="13" y2="10" stroke="#d1d5db" strokeWidth="1.2"/><line x1="7" y1="13" x2="10" y2="13" stroke="#d1d5db" strokeWidth="1.2"/></svg>
                  </div>
                  <div style={{fontSize:15,fontFamily:"var(--di)",fontWeight:800,color:"var(--sub)",marginBottom:6}}>No Report Yet</div>
                  <div style={{fontSize:12,color:"var(--mt)",fontWeight:500}}>Run the agent pipeline to generate the investigation report</div>
                </div>
              ) : (
                <div style={{animation:"fadeIn .4s ease"}}>
                  <div style={{background:rb,border:`1.5px solid ${rbd}`,borderRadius:12,padding:22,marginBottom:18,boxShadow:"var(--shadow-md)"}}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:18}}>
                      <div>
                        <LB c={rc} mb={6}>Fraud Investigation Report</LB>
                        <div style={{fontSize:20,fontFamily:"var(--di)",fontWeight:800,color:"var(--tx)",letterSpacing:"-.01em",marginBottom:3}}>{claim.id||"CLM-NEW"}</div>
                        <div style={{fontSize:12,color:"var(--mt)",fontWeight:600}}>{claim.customer} · Rs {Number(claim.claimAmount).toLocaleString("en-IN")}</div>
                      </div>
                      <div style={{textAlign:"right"}}>
                        <div style={{fontSize:52,fontFamily:"var(--di)",fontWeight:800,color:rc,lineHeight:1}}>{(s*100).toFixed(0)}%</div>
                        <div style={{fontSize:10,fontFamily:"var(--di)",fontWeight:800,color:rc,letterSpacing:".14em"}}>FRAUD RISK</div>
                        <div style={{marginTop:6,fontSize:11,fontFamily:"var(--di)",fontWeight:800,color:rc,background:"white",border:`2px solid ${rbd}`,padding:"5px 14px",borderRadius:6,boxShadow:"var(--shadow)"}}>{rl} · {ra}</div>
                      </div>
                    </div>
                    <div style={{height:1,background:rbd,marginBottom:18}}/>
                    <pre style={{fontSize:12,fontFamily:"var(--mo)",color:"var(--tx)",whiteSpace:"pre-wrap",lineHeight:1.9,fontWeight:500}}>{outputs.report}</pre>
                  </div>
                  <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:7,marginBottom:18}}>
                    {AGENTS.map(a=>(
                      <div key={a.id} style={{background:a.bg,border:`1.5px solid ${a.color}40`,borderRadius:7,padding:"10px 8px",textAlign:"center",boxShadow:"var(--shadow)"}}>
                        <div style={{fontSize:8,fontFamily:"var(--di)",fontWeight:800,color:a.color,letterSpacing:".12em",marginBottom:2}}>COMPLETE</div>
                        <div style={{fontSize:10,color:"var(--mt)",fontWeight:700}}>{a.name}</div>
                      </div>
                    ))}
                  </div>
                  {outputs.reason && (
                    <div style={{background:"var(--s1)",border:"1.5px solid var(--bd)",borderRadius:10,padding:18,boxShadow:"var(--shadow)"}}>
                      <LB mb={12}>Investigator Reasoning · Llama 3 70B</LB>
                      <div style={{fontSize:12,color:"var(--mt)",lineHeight:1.85,fontWeight:500}}>{outputs.reason}</div>
                    </div>
                  )}
                </div>
              )
            )}

            {/* ── AUDIT ── */}
            {tab==="audit" && (<>
              <div style={{marginBottom:18,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                <div>
                  <div style={{fontSize:18,fontFamily:"var(--di)",fontWeight:800,marginBottom:3}}>Audit Log</div>
                  <div style={{fontSize:12,color:"var(--mt)",fontWeight:600}}>{audit.length} claim{audit.length!==1?"s":""} analysed this session</div>
                </div>
                {audit.length>0 && (
                  <div style={{display:"flex",gap:8}}>
                    {["HIGH","MEDIUM","LOW"].map(l=>{
                      const count=audit.filter(e=>e.verdict===l).length;
                      const col=l==="HIGH"?"var(--rd)":l==="MEDIUM"?"var(--am)":"var(--gn)";
                      const bg=l==="HIGH"?"var(--rd-lt)":l==="MEDIUM"?"var(--am-lt)":"var(--gn-lt)";
                      const bdr=l==="HIGH"?"#fecaca":l==="MEDIUM"?"#fde68a":"#bbf7d0";
                      return <div key={l} style={{padding:"5px 12px",background:bg,border:`1.5px solid ${bdr}`,borderRadius:6}}><span style={{fontSize:10,fontFamily:"var(--di)",fontWeight:800,color:col}}>{count} {l}</span></div>;
                    })}
                  </div>
                )}
              </div>
              {audit.length===0 ? (
                <div style={{textAlign:"center",padding:"72px 20px"}}>
                  <div style={{fontSize:15,fontFamily:"var(--di)",fontWeight:800,color:"var(--sub)",marginBottom:6}}>No Claims Analysed Yet</div>
                  <div style={{fontSize:12,color:"var(--mt)",fontWeight:500}}>Run the pipeline on any claim to populate the log</div>
                </div>
              ) : (<>
                <div style={{background:"var(--s1)",border:"1.5px solid var(--bd)",borderRadius:10,overflow:"hidden",boxShadow:"var(--shadow)",marginBottom:16}}>
                  <div style={{display:"grid",gridTemplateColumns:"150px 1fr 75px 115px 80px 65px 60px",padding:"10px 16px",background:"var(--s2)",borderBottom:"1.5px solid var(--bd)"}}>
                    {["CLAIM ID","CUSTOMER","AMOUNT","SCORE","VERDICT","SOURCE","TIME"].map(h=>(
                      <div key={h} style={{fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:"var(--sub)",letterSpacing:".14em"}}>{h}</div>
                    ))}
                  </div>
                  {audit.map((e,i)=>{
                    const col=riskColor(e.score),bg=riskBg(e.score);
                    return (
                      <div key={i} className="hr" style={{display:"grid",gridTemplateColumns:"150px 1fr 75px 115px 80px 65px 60px",padding:"11px 16px",borderBottom:i<audit.length-1?"1.5px solid var(--bd)":"none"}}>
                        <div style={{fontSize:10,fontFamily:"var(--mo)",fontWeight:700,color:"var(--bl)"}}>{e.id}</div>
                        <div style={{fontSize:11,fontFamily:"var(--di)",fontWeight:800}}>{e.customer}</div>
                        <div style={{fontSize:10,fontFamily:"var(--mo)",fontWeight:600,color:"var(--mt)"}}>₹{(Number(e.amount)/1e5).toFixed(1)}L</div>
                        <div>
                          <div style={{height:4,background:"var(--bd)",borderRadius:2,overflow:"hidden",marginBottom:3}}>
                            <div style={{width:`${e.score*100}%`,height:"100%",background:col,borderRadius:2}}/>
                          </div>
                          <span style={{fontSize:10,fontFamily:"var(--mo)",fontWeight:700,color:col}}>{(e.score*100).toFixed(1)}%</span>
                        </div>
                        <div style={{padding:"3px 8px",background:bg,borderRadius:4,fontSize:9,fontFamily:"var(--di)",fontWeight:800,color:col,alignSelf:"center",width:"fit-content",border:`1.5px solid ${col}30`}}>{e.verdict}</div>
                        <div style={{fontSize:9,fontFamily:"var(--di)",fontWeight:700,color:"var(--sub)",alignSelf:"center"}}>{e.source==="upload"?"DOC":e.source==="new"?"FORM":"SAMPLE"}</div>
                        <div style={{fontSize:9,fontFamily:"var(--mo)",fontWeight:600,color:"var(--sub)",alignSelf:"center"}}>{e.time}</div>
                      </div>
                    );
                  })}
                </div>
                <div style={{background:"var(--s1)",border:"1.5px solid var(--bd)",borderRadius:10,padding:18,boxShadow:"var(--shadow)"}}>
                  <LB mb={14}>Session Statistics</LB>
                  <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10}}>
                    {[["Total Analysed",audit.length,"var(--bl)","var(--bl-lt)","#bfdbfe"],["High Risk",audit.filter(e=>e.verdict==="HIGH").length,"var(--rd)","var(--rd-lt)","#fecaca"],["For Review",audit.filter(e=>e.verdict==="MEDIUM").length,"var(--am)","var(--am-lt)","#fde68a"],["Approved",audit.filter(e=>e.verdict==="LOW").length,"var(--gn)","var(--gn-lt)","#bbf7d0"]].map(([l,v,c,bg,bdr])=>(
                      <div key={l} style={{padding:14,background:bg,borderRadius:8,border:`1.5px solid ${bdr}`,textAlign:"center",boxShadow:"var(--shadow)"}}>
                        <div style={{fontSize:30,fontFamily:"var(--di)",fontWeight:800,color:c}}>{v}</div>
                        <div style={{fontSize:10,color:"var(--mt)",marginTop:3,fontWeight:700}}>{l}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </>)}
            </>)}

          </div>
        </div>
      </div>
    </div>
  );
}
