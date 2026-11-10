// Compiles TouristDigitalID.sol, deploys it on an in-memory Ethereum chain (Ganache)
// and tests registration, verification, revocation, incidents and access control.
//
// Run (inside contracts/):   npm install   then   npm test
const fs = require('fs');
const path = require('path');
const solc = require('solc');
const ganache = require('ganache');
const { ethers } = require('ethers');

const src = fs.readFileSync(path.join(__dirname, 'TouristDigitalID.sol'), 'utf8');
const out = JSON.parse(solc.compile(JSON.stringify({
  language: 'Solidity',
  sources: { 'TouristDigitalID.sol': { content: src } },
  settings: { evmVersion: 'paris', optimizer: { enabled: true, runs: 200 }, outputSelection: { '*': { '*': ['abi', 'evm.bytecode'] } } },
})));
const errs = (out.errors || []).filter(e => e.severity === 'error');
if (errs.length) { console.error(errs); process.exit(1); }
const art = out.contracts['TouristDigitalID.sol'].TouristDigitalID;
fs.mkdirSync(path.join(__dirname, 'build'), { recursive: true });
fs.writeFileSync(path.join(__dirname, 'build', 'TouristDigitalID.json'),
  JSON.stringify({ abi: art.abi, bytecode: art.evm.bytecode.object }, null, 1));

let passed = 0, failed = 0;
const check = (name, cond) => { if (cond) { passed++; console.log('  PASS', name); } else { failed++; console.log('  FAIL', name); } };
const reverts = async (p) => { try { await (await p).wait?.(); return false; } catch { return true; } };

(async () => {
  const provider = new ethers.BrowserProvider(ganache.provider({ logging: { quiet: true }, chain: { hardfork: 'shanghai' } }));
  const [tourism, police, outsider] = await Promise.all([0, 1, 2].map(i => provider.getSigner(i)));
  const factory = new ethers.ContractFactory(art.abi, art.evm.bytecode.object, tourism);
  const c = await factory.deploy();
  await c.waitForDeployment();
  console.log('Deployed at', await c.getAddress());

  const h = s => ethers.sha256(ethers.toUtf8Bytes(s));
  const id = h('TID-ABC123');
  const cred = h('credential-json');
  const now = (await provider.getBlock('latest')).timestamp;

  await (await c.addAuthority(await police.getAddress())).wait();
  check('admin added police as authority', await c.isAuthority(await police.getAddress()));

  await (await c.registerTourist(id, cred, h('kyc'), h('itinerary'), now - 3600, now + 7 * 86400)).wait();
  let [ok, why] = await c.verifyTourist(id, cred);
  check('registered ID verifies', ok && why === 'valid');
  [ok, why] = await c.verifyTourist(id, h('forged'));
  check('forged credential rejected', !ok && why === 'credential does not match record');
  check('duplicate registration reverts', await reverts(c.registerTourist(id, cred, h('k'), h('i'), now, now + 1)));
  check('outsider cannot register', await reverts(c.connect(outsider).registerTourist(h('x'), cred, h('k'), h('i'), now, now + 1)));
  check('invalid period reverts', await reverts(c.registerTourist(h('y'), cred, h('k'), h('i'), now + 10, now)));

  await (await c.connect(police).recordIncident(id, 129860, 775958, 3, true, h('alert-1'))).wait();
  check('incident recorded', (await c.incidentCount()) === 1n);
  const inc = await c.getIncident(0);
  check('incident fields stored', inc.risk === 3n && inc.latE4 === 129860n && inc.alertHash === h('alert-1'));
  check('incidents indexed by tourist', (await c.incidentsOfTourist(id)).length === 1);
  check('incident for unknown ID reverts', await reverts(c.recordIncident(h('nobody'), 0, 0, 0, false, h('a'))));

  await (await c.connect(police).revokeTourist(id, 'lost phone')).wait();
  [ok, why] = await c.verifyTourist(id, cred);
  check('revoked ID rejected', !ok && why === 'ID revoked');

  const id2 = h('TID-EXPIRED');
  await (await c.registerTourist(id2, cred, h('k'), h('i'), now - 10 * 86400, now - 86400)).wait();
  [ok, why] = await c.verifyTourist(id2, cred);
  check('expired ID rejected', !ok && why === 'ID not valid today');

  check('non-admin cannot add authority', await reverts(c.connect(police).addAuthority(await outsider.getAddress())));

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
})();
