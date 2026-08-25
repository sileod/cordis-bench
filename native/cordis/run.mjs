import fs from 'node:fs'
import process from 'node:process'
import { Context, FiberState } from 'cordis'

const CORDIS_VERSION = '4.0.0-rc.7'
const CORDIS_SOURCE_COMMIT = '56b3d4f725681cf4556c1a8695a709cc3b6eed74'

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function canonicalObservation(state, fibers) {
  const active = Object.entries(fibers)
    .filter(([, fiber]) => fiber.state === FiberState.ACTIVE)
    .map(([name]) => name)
    .sort()
  return {
    slots: Object.fromEntries(Object.entries(state).sort(([a], [b]) => a.localeCompare(b))),
    active,
  }
}

function normalizeObservation(observation) {
  return {
    slots: Object.fromEntries(
      Object.entries(observation.slots).sort(([a], [b]) => a.localeCompare(b)),
    ),
    active: [...observation.active].sort(),
  }
}

function equalObservation(a, b) {
  return JSON.stringify(normalizeObservation(a)) === JSON.stringify(normalizeObservation(b))
}

function makeProvider(name, service) {
  return {
    name,
    apply(ctx) {
      ctx.provide(service, { provider: name })
    },
  }
}

function makeLeaf(def, state, trace, delayByName) {
  return {
    name: def.name,
    inject: [def.service],
    apply(ctx) {
      ctx.effect(() => {
        const previous = state[def.slot]
        state[def.slot] = def.value
        trace.push({ event: 'start', package: def.name, slot: def.slot, previous, value: def.value })
        return async () => {
          const delay = delayByName[def.name] ?? 0
          if (delay) await sleep(delay)
          const beforeRestore = state[def.slot]
          state[def.slot] = previous
          trace.push({
            event: 'restore',
            package: def.name,
            slot: def.slot,
            previous,
            before_restore: beforeRestore,
            delay_ms: delay,
          })
        }
      }, `cordis-bench:${def.name}`)
    },
  }
}

function makeDecoy(name) {
  return { name, apply() {} }
}

function scheduleDelays(spec, schedule) {
  const delays = {}
  const order = spec.schedules[schedule]
  if (!order) throw new Error(`unknown schedule ${schedule}`)
  order.forEach((name, index) => {
    delays[name] = index * 4
  })
  return delays
}

async function instantiate(spec, schedule) {
  const state = { ...spec.base_slots }
  const trace = []
  const root = new Context()
  root.logger.error = (...args) => {
    trace.push({ event: 'cordis_error', message: args.map(String).join(' ') })
  }

  const delayByName = scheduleDelays(spec, schedule)
  const defs = Object.fromEntries(spec.leaves.map((leaf) => [leaf.name, leaf]))
  const fibers = {}
  const plugins = {}

  for (const provider of spec.providers) {
    plugins[provider.name] = makeProvider(provider.name, provider.service)
    fibers[provider.name] = await root.plugin(plugins[provider.name])
  }
  plugins[spec.decoy] = makeDecoy(spec.decoy)
  fibers[spec.decoy] = await root.plugin(plugins[spec.decoy])

  for (const name of spec.startup_order) {
    const def = defs[name]
    plugins[name] = makeLeaf(def, state, trace, delayByName)
    fibers[name] = await root.plugin(plugins[name])
  }

  return { root, state, trace, fibers }
}

async function executeCommand(runtime, command) {
  runtime.trace.push({ event: 'command', kind: command.kind, package: command.name })
  if (command.kind === 'inspect') return
  if (command.kind !== 'dispose') throw new Error(`unknown command kind ${command.kind}`)
  const fiber = runtime.fibers[command.name]
  if (!fiber) throw new Error(`unknown package ${command.name}`)
  await fiber.dispose()
  await Promise.all(Object.values(runtime.fibers).map((item) => item.await()))
}

async function runPlan(spec, plan, schedule) {
  const runtime = await instantiate(spec, schedule)
  for (const command of plan) await executeCommand(runtime, command)
  const observation = canonicalObservation(runtime.state, runtime.fibers)
  const trace = runtime.trace.slice()
  await runtime.root.fiber.dispose()
  return { observation, trace }
}

async function evaluateSpec(spec) {
  const scheduleRuns = []
  for (const label of Object.keys(spec.schedules)) {
    const run = await runPlan(spec, [{ kind: 'dispose', name: spec.query_provider }], label)
    scheduleRuns.push({ label, ...run })
  }
  const confluent = scheduleRuns.every((run) => equalObservation(run.observation, scheduleRuns[0].observation))

  const optionSuccess = {}
  const optionRuns = {}
  for (const [letter, plan] of Object.entries(spec.options)) {
    const runs = []
    for (const label of Object.keys(spec.schedules)) {
      runs.push({ label, ...(await runPlan(spec, plan, label)) })
    }
    optionRuns[letter] = runs
    optionSuccess[letter] = runs.every((run) => equalObservation(run.observation, spec.target))
  }

  return {
    native_case_id: spec.native_case_id,
    cordis_package: 'cordis',
    cordis_version: CORDIS_VERSION,
    cordis_source_commit: CORDIS_SOURCE_COMMIT,
    node_version: process.version,
    schedule_runs: scheduleRuns,
    confluent,
    option_success: optionSuccess,
    option_runs: optionRuns,
  }
}

async function selfTest() {
  const spec = {
    native_case_id: 'self-test',
    base_slots: { alpha: 'base-a', beta: 'base-b', gamma: 'base-c' },
    providers: [
      { name: 'critical-provider', service: 'critical.service' },
      { name: 'control-provider', service: 'control.service' },
    ],
    decoy: 'decoy',
    leaves: [
      { name: 'a', service: 'critical.service', slot: 'alpha', value: 'A' },
      { name: 'b', service: 'critical.service', slot: 'alpha', value: 'B' },
      { name: 'c', service: 'control.service', slot: 'beta', value: 'C' },
      { name: 'd', service: 'control.service', slot: 'gamma', value: 'D' },
    ],
    startup_order: ['a', 'b', 'c', 'd'],
    query_provider: 'critical-provider',
    schedules: { ab: ['a', 'b'], ba: ['b', 'a'] },
    target: {
      slots: { alpha: 'base-a', beta: 'C', gamma: 'D' },
      active: ['c', 'control-provider', 'd', 'decoy'].sort(),
    },
    options: {
      A: [{ kind: 'inspect', name: 'critical-provider' }, { kind: 'dispose', name: 'critical-provider' }],
      B: [{ kind: 'dispose', name: 'b' }, { kind: 'dispose', name: 'critical-provider' }],
      C: [{ kind: 'dispose', name: 'a' }, { kind: 'dispose', name: 'critical-provider' }],
      D: [{ kind: 'dispose', name: 'decoy' }, { kind: 'dispose', name: 'critical-provider' }],
    },
  }
  const result = await evaluateSpec(spec)
  if (result.confluent) throw new Error('self-test expected non-confluence')
  const good = Object.entries(result.option_success).filter(([, value]) => value).map(([key]) => key)
  if (JSON.stringify(good) !== JSON.stringify(['B'])) {
    throw new Error(`self-test expected only B to succeed, got ${JSON.stringify(good)}`)
  }
  process.stdout.write(JSON.stringify({ ok: true, cordis_version: CORDIS_VERSION, node_version: process.version }) + '\n')
}

async function main() {
  if (process.argv.includes('--self-test')) return selfTest()
  const path = process.argv[2]
  const text = path ? fs.readFileSync(path, 'utf8') : fs.readFileSync(0, 'utf8')
  const specs = text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line))
  for (const spec of specs) {
    const result = await evaluateSpec(spec)
    process.stdout.write(JSON.stringify(result) + '\n')
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
