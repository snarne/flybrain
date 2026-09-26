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
  controls.target.set(0, 0, 0.5);
  controls.enableDamping = true;
  controls.minDistance = 3;
  controls.maxDistance = 120;

  scene.add(new THREE.HemisphereLight(0xcfd8ff, 0x201810, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(4, 6, 10);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 1.2);
  rim.position.set(-8, -4, 3);
  scene.add(rim);
  // a soft light from the viewer, so the side of the fly facing the camera is never in the dark
  // (the fly turns; the lights above don't)
  const head = new THREE.DirectionalLight(0xffffff, 0.9);
  head.position.set(0, 0, 1);
  camera.add(head);
  scene.add(camera);

  // ---------------------------------------------------------------- the arena: floor, walls, stripes
  const ARENA = 40;                                  // half-width (mm), as server/physics.py
  const floorTex = (() => {
    const c = document.createElement('canvas');
    c.width = c.height = 512;
    const g = c.getContext('2d');
    g.fillStyle = '#1b2029';
    g.fillRect(0, 0, 512, 512);
    g.strokeStyle = '#2a3140';
    g.lineWidth = 2;
    for (let i = 0; i <= 512; i += 32) {           // 1 mm grid (texture repeats every 16 mm)
      g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 512); g.stroke();
      g.beginPath(); g.moveTo(0, i); g.lineTo(512, i); g.stroke();
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(2 * ARENA / 16, 2 * ARENA / 16);
    t.anisotropy = 4;
    return t;
  })();
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(2 * ARENA, 2 * ARENA), new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.95 }));
  scene.add(floor);
  // the arena's walls: glass, as tall as the fly can fly (server/physics.py keeps it inside them)
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x3a4458, roughness: 0.3, transparent: true, opacity: 0.18, depthWrite: false });
  for (const [x, y, w, h] of [[0, ARENA, 2 * ARENA, 0.4], [0, -ARENA, 2 * ARENA, 0.4], [ARENA, 0, 0.4, 2 * ARENA], [-ARENA, 0, 0.4, 2 * ARENA]]) {
    const wall = new THREE.Mesh(new THREE.BoxGeometry(w, h, 16), wallMat);
    wall.position.set(x, y, 8);
    scene.add(wall);
  }
  // a striped drum around the arena, shown while the stripes turn
  const stripeTex = (() => {
    const c = document.createElement('canvas');
    c.width = 1024; c.height = 8;
    const g = c.getContext('2d');
    for (let i = 0; i < 24; i++) { g.fillStyle = i % 2 ? '#0c0e12' : '#d8dde8'; g.fillRect(i * 1024 / 24, 0, 1024 / 24, 8); }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  })();
  const drum = new THREE.Mesh(new THREE.CylinderGeometry(ARENA * 1.2, ARENA * 1.2, 30, 96, 1, true),
    new THREE.MeshBasicMaterial({ map: stripeTex, side: THREE.BackSide, transparent: true, opacity: 0.9 }));
  drum.rotation.x = Math.PI / 2;
  drum.position.z = 10;
  drum.visible = false;
  scene.add(drum);
  // things in the world, drawn from the server's arena (server/arena.py)
  const worldGroup = new THREE.Group();
  scene.add(worldGroup);
  const worldObjs = new Map();
  const ODOUR_COL = { fruit: 0xffa040, co2: 0x9aa4b2, geosmin: 0x7a8a3a, cva: 0xff7ab8 };
  const ZONE_COL = { heat: 0xff4d3a, cold: 0x3a8cff, humid: 0x3ad6d0, dry: 0xd8b060 };
  function radialTexture(hex) {
    const c = document.createElement('canvas');
    c.width = c.height = 128;
    const g = c.getContext('2d');
    const col = new THREE.Color(hex);
    const rgb = `${Math.round(col.r * 255)},${Math.round(col.g * 255)},${Math.round(col.b * 255)}`;
    const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    gr.addColorStop(0, `rgba(${rgb},0.55)`);
    gr.addColorStop(0.6, `rgba(${rgb},0.18)`);
    gr.addColorStop(1, `rgba(${rgb},0)`);
    g.fillStyle = gr;
    g.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  }
  function makeObj(o) {
    const g = new THREE.Group();
    if (o.kind === 'sugar' || o.kind === 'bitter') {
      const m = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2),
        new THREE.MeshPhysicalMaterial({ color: o.kind === 'sugar' ? 0xffc050 : 0x9b6bff, roughness: 0.05, transmission: 0.3, transparent: true, opacity: 0.85 }));
      m.rotation.x = Math.PI / 2;
      m.scale.set(1, 0.35, 1);
      g.add(m);
      g.userData.drop = m;
    } else if (o.kind === 'odour') {
      const plume = new THREE.Mesh(new THREE.CircleGeometry(2.2 * o.spread, 64),
        new THREE.MeshBasicMaterial({ map: radialTexture(ODOUR_COL[o.odour] || 0xffffff), transparent: true, depthWrite: false }));
      plume.position.z = 0.02;
      const vial = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 1.6, 24),
        new THREE.MeshStandardMaterial({ color: ODOUR_COL[o.odour] || 0xffffff, roughness: 0.3, metalness: 0.1 }));
      vial.rotation.x = Math.PI / 2;
      vial.position.z = 0.8;
      g.add(plume, vial);
    } else if (ZONE_COL[o.kind]) {
      const disc = new THREE.Mesh(new THREE.CircleGeometry(o.r + 1.5, 64),
        new THREE.MeshBasicMaterial({ map: radialTexture(ZONE_COL[o.kind]), transparent: true, depthWrite: false }));
      disc.position.z = 0.01;
      g.add(disc);
    } else if (o.kind === 'target') {
      const m = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshStandardMaterial({ color: 0x14100c, roughness: 0.7 }));
      m.scale.set(o.r * 1.6, o.r, o.r * 0.8);
      g.add(m);
    } else if (o.kind === 'threat') {
      g.add(new THREE.Mesh(new THREE.SphereGeometry(o.r, 32, 20), new THREE.MeshStandardMaterial({ color: 0x050608, roughness: 0.9 })));
    }
    worldGroup.add(g);
    return g;
  }
  function syncWorld(A) {
    const seen = new Set();
    for (const o of A?.objects || []) {
      seen.add(o.id);
      let g = worldObjs.get(o.id);
      if (!g) { g = makeObj(o); worldObjs.set(o.id, g); }
      g.position.set(o.x, o.y, o.kind === 'threat' || o.kind === 'target' ? o.z : 0);
      if (g.userData.drop) g.userData.drop.scale.set(o.r, 0.35 * o.r, o.r);
    }
    for (const [id, g] of worldObjs) if (!seen.has(id)) { worldGroup.remove(g); worldObjs.delete(id); }
    drum.visible = !!A?.stripes;
    if (A?.stripes) drum.rotation.y = A.stripe_angle;
  }

  // ---------------------------------------------------------------- the fly
  const fly = new THREE.Group();
  scene.add(fly);
  const nodes = {};
  const mats = {};
  const baseQuat = {};
  const EYE_GLOW = 0x2a0503;     // compound eyes have their own red sheen, whichever way the light falls
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
  // resting pose: the middle of the range each joint uses in recorded walking (server/motor.py uses
  // the same centre, and sends offsets from it)
  const neutral = Object.fromEntries(LEGS.map(l => [l, Object.fromEntries(STEP_DOFS.map(d => {
    const a = S.legs[l][d];
    return [d, (Math.min(...a) + Math.max(...a)) / 2];
  }))]));

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
  let st = { mode: 'rest', joints: null, parts: {}, ball: [0, 0], probe: null, dust: false, fly: null, arena: null };
  let t = 0, wingPhase = 0, flyPlaced = false;
  const cur = {};   // smoothed joint offsets
  function onBody(s) { st = s; }

  const matsWhere = re => Object.entries(mats).filter(([n]) => re.test(n)).map(([, m]) => m);
  const glowGroups = {
    ...Object.fromEntries(LEGS.map(l => [l, matsWhere(new RegExp(`^${l}(Coxa|Femur|Tibia|Tarsus)`))])),
    wingL: [mats.LWing].filter(Boolean), wingR: [mats.RWing].filter(Boolean),
    haltereL: [mats.LHaltere].filter(Boolean), haltereR: [mats.RHaltere].filter(Boolean),
    abdomen: ['A1A2', 'A3', 'A4', 'A5', 'A6'].map(n => mats[n]).filter(Boolean),
    proboscis: [mats.Rostrum, mats.Haustellum].filter(Boolean),
    antennaL: matsWhere(/^L(Antenna|Pedicel|Funiculus|Arista)/), antennaR: matsWhere(/^R(Antenna|Pedicel|Funiculus|Arista)/),
  };
  let glowOn = true;
  const sm = (key, target, k) => (cur[key] = (cur[key] ?? target) + (target - (cur[key] ?? target)) * k);

  // wing orientation, built from scratch in the thorax frame:
  //   base: mesh span pointing out sideways (the left wing's mesh is the right one mirrored, so it
  //   already points left), membrane horizontal;
  //   then pitch about the span (wing rotation), elevation, and stroke/fold about the vertical
  const qBase = { L: new THREE.Quaternion(), R: new THREE.Quaternion() };
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), qc = new THREE.Quaternion();
  function wingQuat(side, stroke, elev, pitch, out) {
    const s = side === 'L' ? 1 : -1;
    qa.setFromAxisAngle(Z, s * stroke);
    qb.setFromAxisAngle(X, s * elev);
    qc.setFromAxisAngle(Y, pitch);
    return out.copy(qa).multiply(qb).multiply(qc).multiply(qBase[side]);
  }
  const FOLD = 1.66, MID = 0.35;
  const qRoll = new THREE.Quaternion();         // radians: folded back over the abdomen / middle of the flight stroke

  function update(dt) {
    t += dt;
    const J = st.joints;
    const P = st.physics || {};
    // smooth the 30 Hz joint stream (the physical body's poses are already smooth: follow them closely)
    const k = Math.min(1, dt * (P.on ? 25 : 10));
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
    // flight power: the wing power motor neurons', or the flight kept up once airborne (server/body.py)
    const power = sm('power', Math.max(W.L?.power || 0, st.airborne ? (st.flight_power || 0) : 0), Math.min(1, dt * 6));
    const beating = !!st.airborne && power > 0.12;   // feet on the ground inhibit flight (tarsal reflex)
    wingPhase += dt * 2 * Math.PI * (beating ? 7 + 5 * Math.min(1, power) : 0);   // shown ~25x slower than 200 Hz
    for (const side of ['L', 'R']) {
      const w = W[side] || {};
      // posture muscles only lift the wing out of its folded position when clearly active
      const spread = sm('spread' + side, Math.min(1, Math.max(Math.max(0, ((w.extend || 0) - 0.3) / 0.7), beating ? 1 : 0)), Math.min(1, dt * 5));
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
    // halteres: tilted by their own motor neurons, and beating in antiphase with the wings in flight
    const Hl = J?.haltere || {};
    for (const side of ['L', 'R']) extra(side + 'Haltere', Y, sm('hal' + side, 0.5 * (Hl[side] || 0), k) + (beating ? 0.6 * Math.cos(wingPhase + Math.PI) : 0));
    // abdomen: segmental motor neurons curl it down, one side more than the other bends it sideways
    const Ab = J?.abdomen || {};
    const curl = sm('curl', Ab.curl || 0, k), bend = sm('bend', Ab.bend || 0, k);
    ['A1A2', 'A3', 'A4', 'A5', 'A6'].forEach((seg, i) => {
      const n = nodes[seg];
      if (!n) return;
      const q = new THREE.Quaternion().setFromAxisAngle(Y, -0.1 * curl * (i ? 1 : 0.4));
      q.multiply(new THREE.Quaternion().setFromAxisAngle(Z, 0.08 * bend * (i ? 1 : 0.4)));
      n.userData.extra = q;
    });
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
    for (const [name, mat] of Object.entries(mats)) mat.emissive.setHex(/Eye/.test(name) ? EYE_GLOW : 0);
    if (glowOn) {
      const P = st.parts || {};
      for (const [key, list] of Object.entries(glowGroups)) {
        const v = Math.min(1, P[key] || 0);
        if (v > 0.12) for (const mat of list) mat.emissive.lerp(HOT, (v - 0.1) * 0.4);
      }
    }

    // where the fly is: its root body's position and orientation from the physics (or the kinematic
    // body's walk across the floor); the camera follows it
    const pose = st.fly;
    if (pose?.pos) {
      const [x, y, z] = pose.pos;
      const target = new THREE.Vector3(x, y, z ?? 0);
      if (!flyPlaced || target.distanceTo(fly.position) > 20) { fly.position.copy(target); flyPlaced = true; }
      else fly.position.lerp(target, Math.min(1, dt * 20));
      const q = pose.quat;
      fly.quaternion.slerp(new THREE.Quaternion(q[1], q[2], q[3], q[0]), Math.min(1, dt * 20));
      const follow = new THREE.Vector3(fly.position.x, fly.position.y, Math.max(0.35, fly.position.z + 0.9));   // the thorax
      const shift = follow.clone().sub(controls.target);
      controls.target.add(shift);
      camera.position.add(shift);
    }
    syncWorld(st.arena);

    // the experimenter's props: a drop on a needle touched to the proboscis, dust on the antennae
    const h = st.fly?.heading || 0, ch = Math.cos(h), sh = Math.sin(h);
    if (st.probe) {
      probe.visible = true;
      dropMat.color.set(st.probe.kind === 'sugar' ? 0xffb640 : 0x9b6bff);
      const tip = new THREE.Vector3();
      (nodes.Haustellum || nodes.Head).getWorldPosition(tip);
      const pt = st.probe.t;
      const approach = pt < 0.35 ? 1 - pt / 0.35 : pt > 4.2 ? Math.min(1, (pt - 4.2) / 0.4) : 0;
      const d = 0.35 + 2.5 * approach;
      probe.position.set(tip.x + d * ch, tip.y + d * sh, tip.z - 0.25);
      probe.rotation.z = h;
    } else probe.visible = false;
    dust.visible = !!st.dust;
    if (st.dust) {
      const head = new THREE.Vector3();
      nodes.Head.getWorldPosition(head);
      dustSeed.forEach(([a, b, c], i) => {
        const r = 0.3 + 0.7 * ((a + t * 0.3) % 1);
        const fx = 0.4 + r * Math.cos(b * 6.28 + t), fy = (c - 0.5) * 1.2;
        dustPos[i * 3] = head.x + fx * ch - fy * sh;
        dustPos[i * 3 + 1] = head.y + fx * sh + fy * ch;
        dustPos[i * 3 + 2] = head.z + 0.25 + r * Math.sin(b * 6.28 + t) * 0.5;
      });
      dustGeo.attributes.position.needsUpdate = true;
    }
    controls.update();
  }

  // placing things by clicking the floor: where a point on the view lands on the arena floor
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), hit = new THREE.Vector3();
  const floorPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  function pickFloor(clientX, clientY) {
    const r = dom.getBoundingClientRect();
    ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    if (!ray.ray.intersectPlane(floorPlane, hit)) return null;
    if (Math.abs(hit.x) > ARENA || Math.abs(hit.y) > ARENA) return null;
    return { x: hit.x, y: hit.y };
  }
  const marker = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.2, 40),
    new THREE.MeshBasicMaterial({ color: 0xffc857, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide }));
  marker.position.z = 0.03;
  marker.visible = false;
  scene.add(marker);
  function showMarker(p) {
    marker.visible = !!p;
    if (p) marker.position.set(p.x, p.y, 0.03);
  }

  return { scene, camera, controls, update, onBody, nodes, setGlow: v => { glowOn = v; }, pickFloor, showMarker };
}
