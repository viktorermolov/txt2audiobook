const assert = require('node:assert/strict')
const fs = require('node:fs')
const { createRequire } = require('node:module')
const path = require('node:path')

const appRequire = createRequire('/app/index.js')
const version = (load, name) => {
  let directory = path.dirname(load.resolve(name))
  while (directory !== path.dirname(directory)) {
    const packageFile = path.join(directory, 'package.json')
    if (fs.existsSync(packageFile)) {
      const metadata = JSON.parse(fs.readFileSync(packageFile, 'utf8'))
      if (metadata.name === name) return metadata.version
    }
    directory = path.dirname(directory)
  }
  throw new Error(`No installed package metadata for ${name}`)
}
const expectVersion = (load, name, expected) => {
  assert.equal(version(load, name), expected, `${name} resolved to the wrong version`)
}

assert.equal(appRequire('./package.json').version, '2.37.0')
// Since 2.37.0 the server is compiled from TypeScript into dist-server/.
assert(fs.statSync('/app/dist-server/index.js').isFile())
assert.equal(process.versions.node.split('.')[0], '24')
assert.equal(process.env.NODE_ENV, 'production')

const expected = {
  'socket.io': '4.8.3',
  'socket.io-adapter': '2.5.8',
  'socket.io-parser': '4.2.7',
  'engine.io': '6.6.10',
  ws: '8.21.3',
  express: '4.22.3',
  'body-parser': '1.20.8',
  'express-session': '1.19.0',
  sequelize: '6.37.8',
  sqlite3: '5.1.7'
}
for (const [name, wanted] of Object.entries(expected)) {
  expectVersion(appRequire, name, wanted)
}

const fromSocket = createRequire(appRequire.resolve('socket.io'))
const fromAdapter = createRequire(fromSocket.resolve('socket.io-adapter'))
const fromEngine = createRequire(fromSocket.resolve('engine.io'))
const fromExpress = createRequire(appRequire.resolve('express'))
expectVersion(fromSocket, 'socket.io-parser', '4.2.7')
expectVersion(fromSocket, 'engine.io', '6.6.10')
expectVersion(fromAdapter, 'ws', '8.21.3')
expectVersion(fromEngine, 'ws', '8.21.3')
expectVersion(fromExpress, 'body-parser', '1.20.8')
// proxy-addr < 2.0.8: IP spoofing via IPv4-mapped IPv6 trust subnets.
expectVersion(fromExpress, 'proxy-addr', '2.0.8')
// moment < 2.31.0: path traversal via non-string locale (CVE-2026-17495).
const fromSequelize = createRequire(appRequire.resolve('sequelize'))
expectVersion(fromSequelize, 'moment', '2.31.0')
expectVersion(createRequire(fromSequelize.resolve('moment-timezone')), 'moment', '2.31.0')

const parser = fromSocket('socket.io-parser')
assert.throws(() => new parser.Decoder().add('50-["audit"]'))
const decoder = new parser.Decoder()
let decoded
decoder.on('decoded', value => { decoded = value })
decoder.add('2["audit",{"valid":true}]')
assert.equal(decoded.data[1].valid, true)

const extension = '/usr/local/lib/nusqlite3/libnusqlite3.so'
assert(fs.statSync(extension).isFile())
const sqlite3 = appRequire('sqlite3')
const db = new sqlite3.Database(':memory:')
db.serialize(() => {
  db.loadExtension(extension, error => {
    if (error) throw error
  })
  db.run('CREATE TABLE hardening (value TEXT NOT NULL)')
  db.run('INSERT INTO hardening (value) VALUES (?)', ['verified'])
  db.get('SELECT value FROM hardening', (error, row) => {
    if (error) throw error
    assert.equal(row.value, 'verified')
  })
  db.close(error => {
    if (error) throw error
    console.log('ABS, Node, consumer dependency graph, Socket.IO, and native SQLite verified')
  })
})
