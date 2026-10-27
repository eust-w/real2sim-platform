import { createGenesisPackage } from '../server/genesisPackage.js'
import { importExample } from '../server/examples.js'

const task = await importExample()
const exported = await createGenesisPackage(task)
console.log(exported.packageDir)
