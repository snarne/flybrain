// The fly body (NeuroMechFly, Apache-2.0). Every joint angle comes from the server's motor model:
// motor neuron firing -> muscle activation -> joint (server/motor.py). Nothing here is scripted except
// the wingbeat waveform itself (a 200 Hz asynchronous-muscle oscillation, shown slowed down), whose
// power, amplitude and posture come from the wing motor neurons.
import * as THREE from './vendor/three.module.min.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'];
const STEP_DOFS = ['Coxa', 'Coxa_roll', 'Coxa_yaw', 'Femur', 'Femur_roll', 'Tibia', 'Tarsus1'];
const HOT = new THREE.Color(1.0, 0.55, 0.15);

export async function createFlyView(dom) {
  const [meta, buf] = await Promise.all([
    fetch('data/fly_body.json').then(r => r.json()),
    fetch('data/fly_body.bin').then(r => r.arrayBuffer()),
  ]);
  const V = new Float32Array(buf, 0, meta.nv * 3);
  const F = new Uint32Array(buf, meta.nv * 12, meta.nf * 3);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0d13);
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 500);
  camera.up.set(0, 0, 1);
  camera.position.set(6.5, 7.5, 4.2);
  const controls = new OrbitControls(camera, dom);
  controls.target.set(-0.4, 0, 0.9);
  controls.enableDamping = true;
  controls.minDistance = 3;
  controls.maxDistance = 40;

  scene.add(new THREE.HemisphereLight(0xcfd8ff, 0x201810, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(4, 6, 10);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 1.2);
  rim.position.set(-8, -4, 3);
  scene.add(rim);

  // ---------------------------------------------------------------- the rig: ball and cup
  const ballR = 4.5;
  const tex = (() => {
    const c = document.createElement('canvas');
    c.width = 512; c.height = 256;
    const g = c.getContext('2d');
    for (let i = 0; i < 16; i++) for (let j = 0; j < 8; j++) {
      g.fillStyle = (i + j) % 2 ? '#2a3140' : '#3a4356';
      g.fillRect(i * 32, j * 32, 32, 32);
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  })();
  const ballPivot = new THREE.Group();
  ballPivot.position.set(-0.45, 0, -ballR + 0.02);
  const ball = new THREE.Mesh(new THREE.SphereGeometry(ballR, 48, 32), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.8 }));
  ballPivot.add(ball);
  scene.add(ballPivot);
  const cup = new THREE.Mesh(new THREE.CylinderGeometry(3.8, 2.6, 3.2, 40, 1, true),
    new THREE.MeshStandardMaterial({ color: 0x3b4150, metalness: 0.6, roughness: 0.35, side: THREE.DoubleSide }));
  cup.rotation.x = Math.PI / 2;
  cup.position.set(-0.45, 0, -ballR - 2.2);
  scene.add(cup);

  // ---------------------------------------------------------------- the fly
  const fly = new THREE.Group();
  scene.add(fly);
  const nodes = {};
  const mats = {};
  const baseQuat = {};
  function colourFor(name) {
    if (/Eye/.test(name)) return { color: 0x9c1b12, roughness: 0.35, metalness: 0.1 };
    if (/Wing/.test(name)) return { color: 0xd9e4ff, roughness: 0.2, transparent: true, opacity: 0.28, side: THREE.DoubleSide, depthWrite: false };
    if (/Tarsus|Tibia|Arista/.test(name)) return { color: 0x3b2a1c, roughness: 0.6 };
    if (/Femur|Coxa|Haltere|Antenna|Pedicel|Funiculus/.test(name)) return { color: 0x6b4a2b, roughness: 0.55 };
    if (/^A\d|A1A2/.test(name)) return { color: 0x6e5436, roughness: 0.5 };
    if (/Rostrum|Haustellum/.test(name)) return { color: 0x8a6a45, roughness: 0.5 };
    return { color: 0x80603c, roughness: 0.45 };
  }
  for (const b of meta.bodies) {
    const node = new THREE.Group();
    node.name = b.name;
    node.position.fromArray(b.pos);
    const q = new THREE.Quaternion(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
    node.quaternion.copy(q);
    baseQuat[b.name] = q;
    node.userData.joints = b.joints.map(j => ({ name: j.name, axis: new THREE.Vector3().fromArray(j.axis).normalize(), angle: 0 }));
    if (b.mesh) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(V.slice(b.mesh.v[0] * 3, (b.mesh.v[0] + b.mesh.v[1]) * 3), 3));
      g.setIndex(new THREE.BufferAttribute(F.slice(b.mesh.f[0] * 3, (b.mesh.f[0] + b.mesh.f[1]) * 3), 1));
      g.computeVertexNormals();
      const m = new THREE.MeshStandardMaterial({ ...colourFor(b.name), emissive: 0x000000 });
      mats[b.name] = m;
      const mesh = new THREE.Mesh(g, m);
      mesh.position.fromArray(b.mesh.pos);
      mesh.quaternion.set(b.mesh.quat[1], b.mesh.quat[2], b.mesh.quat[3], b.mesh.quat[0]);
      node.add(mesh);
    }
    nodes[b.name] = node;
    (b.parent && nodes[b.parent] ? nodes[b.parent] : fly).add(node);
  }
  // ghost wings: faint copies at the ends of the stroke to suggest a 200 Hz wingbeat
  const ghosts = [];
  for (const side of ['L', 'R']) {
    const w = nodes[side + 'Wing'];
    if (!w) continue;
    for (let k = 0; k < 3; k++) {
      const gnode = new THREE.Group();
      gnode.position.copy(w.position);
      const mesh = w.children[0].clone();
      mesh.material = new THREE.MeshBasicMaterial({ color: 0xcfe0ff, transparent: true, opacity: 0.0, side: THREE.DoubleSide, depthWrite: false });
      gnode.add(mesh);
      w.parent.add(gnode);
      ghosts.push({ side, k, node: gnode });
    }
  }

  // ---------------------------------------------------------------- recorded steps (NeuroMechFly)
  const S = meta.steps;
  function stepAngles(leg, phase) {
    const n = S.n;
    const x = ((phase % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) / (2 * Math.PI) * (n - 1);
    const i = Math.floor(x), f = x - i, j = (i + 1) % n;
    const out = {};
    for (const d of STEP_DOFS) {
      const a = S.legs[leg][d];
      out[d] = a[i] * (1 - f) + a[j] * f;
    }
    return out;
  }
  const neutral = Object.fromEntries(LEGS.map(l => [l, stepAngles(l, Math.PI)]));

  const jmap = {};
  for (const n of Object.values(nodes)) for (const j of n.userData.joints) jmap[j.name] = j;
  // setJoint('LFCoxa', '_roll', v) sets joint_LFCoxa_roll; setJoint('LFFemur', '', v) sets joint_LFFemur
  function setJoint(body, suffix, value) {
    const j = jmap[`joint_${body}${suffix}`];
    if (j) j.angle = value;
  }
  function applyJoints() {
    const tmp = new THREE.Quaternion();
    for (const [name, n] of Object.entries(nodes)) {
      if (!n.userData.joints.length && !n.userData.extra) continue;
      n.quaternion.copy(baseQuat[name]);
      for (const j of n.userData.joints) n.quaternion.multiply(tmp.setFromAxisAngle(j.axis, j.angle));
      if (n.userData.extra) n.quaternion.multiply(n.userData.extra);
    }
  }
  function extra(name, axis, angle) {
    const n = nodes[name];
    if (!n) return;
    n.userData.extra = (n.userData.extra || new THREE.Quaternion()).setFromAxisAngle(axis, angle);
  }
  const X = new THREE.Vector3(1, 0, 0), Y = new THREE.Vector3(0, 1, 0), Z = new THREE.Vector3(0, 0, 1);

  // ---------------------------------------------------------------- stimuli props
  const loomObj = new THREE.Mesh(new THREE.SphereGeometry(3, 32, 20), new THREE.MeshStandardMaterial({ color: 0x050608, roughness: 0.9 }));
  loomObj.visible = false;
  scene.add(loomObj);
  const probe = new THREE.Group();
  const needle = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 6, 8), new THREE.MeshStandardMaterial({ color: 0xbfc6d4, metalness: 0.9, roughness: 0.25 }));
  needle.rotation.z = Math.PI / 2;
  needle.position.x = 3.2;
  const dropMat = new THREE.MeshPhysicalMaterial({ color: 0xffb640, roughness: 0.05, transmission: 0.4, transparent: true, opacity: 0.85 });
  const drop = new THREE.Mesh(new THREE.SphereGeometry(0.28, 24, 16), dropMat);
  probe.add(needle, drop);
  probe.visible = false;
  scene.add(probe);
  const dustGeo = new THREE.BufferGeometry();
  const dustN = 160;
  const dustPos = new Float32Array(dustN * 3);
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
  const dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: 0xd8cfb8, size: 0.06, transparent: true, opacity: 0.8 }));
  dust.visible = false;
  scene.add(dust);
  const dustSeed = Array.from({ length: dustN }, () => [Math.random(), Math.random(), Math.random()]);

  // ---------------------------------------------------------------- state from the server
  let st = { mode: 'rest', joints: null, parts: {}, ball: [0, 0], loom: null, probe: null, dust: false };
  let t = 0, wingPhase = 0;
  const cur = {};   // smoothed joint offsets
  function onBody(s) { st = s; }

  const matsWhere = re => Object.entries(mats).filter(([n]) => re.test(n)).map(([, m]) => m);
  const glowGroups = {
    ...Object.fromEntries(LEGS.map(l => [l, matsWhere(new RegExp(`^${l}(Coxa|Femur|Tibia|Tarsus)`))])),
    wingL: [mats.LWing, mats.Thorax].filter(Boolean), wingR: [mats.RWing, mats.Thorax].filter(Boolean),
    head: [mats.Head].filter(Boolean),
    proboscis: [mats.Rostrum, mats.Haustellum].filter(Boolean),
    antennaL: matsWhere(/^L(Antenna|Pedicel|Funiculus|Arista)/), antennaR: matsWhere(/^R(Antenna|Pedicel|Funiculus|Arista)/),
  };
  let glowOn = true;
  const sm = (key, target, k) => (cur[key] = (cur[key] ?? target) + (target - (cur[key] ?? target)) * k);

  // wing orientation, built from scratch in the thorax frame:
  //   base: mesh span (local -Y) pointing out sideways, membrane horizontal
  //   then pitch about the span (wing rotation), elevation, and stroke/fold about the vertical
  const qBase = { L: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI), R: new THREE.Quaternion() };
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), qc = new THREE.Quaternion();
  function wingQuat(side, stroke, elev, pitch, out) {
    const s = side === 'L' ? 1 : -1;
    qa.setFromAxisAngle(Z, s * stroke);
    qb.setFromAxisAngle(X, s * elev);
    qc.setFromAxisAngle(Y, pitch);
    return out.copy(qa).multiply(qb).multiply(qc).multiply(qBase[side]);
  }
  const FOLD = 1.66, MID = 0.35;         // radians: folded back over the abdomen / middle of the flight stroke

  function update(dt) {
    t += dt;
    const J = st.joints;
    const k = Math.min(1, dt * 18);
    // legs: resting pose + offsets computed from muscle activation
    for (const leg of LEGS) {
      const off = J?.legs?.[leg] || {};
      const n = neutral[leg];
      for (const d of STEP_DOFS) {
        const [part, sub] = d.split('_');
        const v = sm(leg + d, n[d] + (off[d] || 0), k);
        setJoint(leg + part, sub ? '_' + sub : '', v);
      }
    }
    // wings: posture from the steering/posture muscles, beat from the power muscles
    const W = J?.wings || { L: {}, R: {} };
    const power = sm('power', W.L?.power || 0, Math.min(1, dt * 6));
    const beating = power > 0.12;
    wingPhase += dt * 2 * Math.PI * (beating ? 7 + 5 * Math.min(1, power) : 0);   // shown ~25x slower than 200 Hz
    for (const side of ['L', 'R']) {
      const w = W[side] || {};
      const spread = sm('spread' + side, Math.min(1, Math.max(w.extend || 0, beating ? 1 : 0)), Math.min(1, dt * 5));
      const amp = beating ? Math.min(1.25, 0.55 + 0.6 * power + 0.35 * (w.stroke || 0)) : 0;
      const rest = FOLD + (MID - FOLD) * spread;
      const node = nodes[side + 'Wing'];
      if (!node) continue;
      const ph = wingPhase;
      const stroke = rest + amp * Math.cos(ph);                 // + = backward
      const pitch = beating ? 0.9 * Math.tanh(3 * Math.sin(ph)) : 0;   // wing flips at each stroke reversal
      const elev = beating ? 0.12 * Math.sin(2 * ph) : -0.06 * (1 - spread);
      node.userData.override = wingQuat(side, stroke, elev, pitch, node.userData.override || new THREE.Quaternion());
      for (const gh of ghosts.filter(g => g.side === side)) {
        const gp = ph - (gh.k + 1) * 0.35;
        wingQuat(side, rest + amp * Math.cos(gp), 0.12 * Math.sin(2 * gp), 0.9 * Math.tanh(3 * Math.sin(gp)), gh.node.quaternion);
        gh.node.children[0].material.opacity = beating ? 0.1 - gh.k * 0.025 : 0;
      }
    }
    // halteres beat in antiphase with the wings
    for (const side of ['L', 'R']) extra(side + 'Haltere', Y, beating ? 0.6 * Math.cos(wingPhase + Math.PI) : 0);
    // head: neck motor neurons
    const H = J?.head || {};
    setJoint('Head', '_yaw', sm('hy', 0.45 * (H.yaw || 0), k));
    setJoint('Head', '', sm('hp', -0.35 * (H.pitch || 0), k));
    setJoint('Head', '_roll', sm('hr', 0.3 * (H.roll || 0), k));
    // proboscis: rostrum protractor (MN9), haustellum extensors, pump
    const Pb = J?.proboscis || {};
    const ext = sm('ros', Math.max(0, (Pb.rostrum || 0) - 0.5 * (Pb.retract || 0)), k);
    extra('Rostrum', Y, -0.9 * ext);
    extra('Haustellum', Y, sm('hau', 1.0 * Math.max(Pb.haustellum || 0, 0.6 * ext), k) + 0.08 * (Pb.pump || 0) * Math.sin(t * 2 * Math.PI * 6));
    // antennae
    const A = J?.antenna || {};
    extra('LPedicel', Y, -0.45 * sm('al', A.L || 0, k));
    extra('RPedicel', Y, -0.45 * sm('ar', A.R || 0, k));
    applyJoints();
    for (const side of ['L', 'R']) {
      const n = nodes[side + 'Wing'];
      if (n?.userData.override) n.quaternion.copy(n.userData.override);
    }

    // muscle glow
    for (const mat of Object.values(mats)) mat.emissive.setRGB(0, 0, 0);
    if (glowOn) {
      const P = st.parts || {};
      for (const [key, list] of Object.entries(glowGroups)) {
        const v = Math.min(1, P[key] || 0);
        if (v > 0.03) for (const mat of list) mat.emissive.lerp(HOT, v * 0.6);
      }
    }

    // ball turns under walking legs
    const ball_ = st.ball || [0, 0];
    ball.rotation.y += (ball_[0] / ballR) * dt * -1;

    // stimuli
    if (st.loom) {
      loomObj.visible = true;
      const sgn = st.loom.side === 'left' ? 1 : -1;
      loomObj.position.set(0.8, sgn * (st.loom.d + 1.5), 1.2 + st.loom.d * 0.25);
    } else loomObj.visible = false;
    if (st.probe) {
      probe.visible = true;
      dropMat.color.set(st.probe.kind === 'sugar' ? 0xffb640 : 0x9b6bff);
      const tip = new THREE.Vector3();
      (nodes.Haustellum || nodes.Head).getWorldPosition(tip);
      const pt = st.probe.t;
      const approach = pt < 0.35 ? 1 - pt / 0.35 : pt > 4.2 ? Math.min(1, (pt - 4.2) / 0.4) : 0;
      probe.position.set(tip.x + 0.35 + 2.5 * approach, tip.y, tip.z - 0.25);
    } else probe.visible = false;
    dust.visible = !!st.dust;
    if (st.dust) {
      const head = new THREE.Vector3();
      nodes.Head.getWorldPosition(head);
      dustSeed.forEach(([a, b, c], i) => {
        const r = 0.3 + 0.7 * ((a + t * 0.3) % 1);
        dustPos[i * 3] = head.x + 0.4 + r * Math.cos(b * 6.28 + t);
        dustPos[i * 3 + 1] = head.y + (c - 0.5) * 1.2;
        dustPos[i * 3 + 2] = head.z + 0.25 + r * Math.sin(b * 6.28 + t) * 0.5;
      });
      dustGeo.attributes.position.needsUpdate = true;
    }
    controls.update();
  }

  return { scene, camera, controls, update, onBody, nodes, setGlow: v => { glowOn = v; } };
}
