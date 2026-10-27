import dotenv from 'dotenv'

// A demo must never inherit the user's paid provider key or private task history.
if (process.env.REAL2SIM_DEMO !== '1') {
  dotenv.config({ path: '.env.local', quiet: true })
  dotenv.config({ quiet: true })
}
