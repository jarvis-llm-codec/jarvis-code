#!/usr/bin/env node
import { APP_TITLE, VERSION } from "../config.js";

process.title = `${APP_TITLE} v${VERSION}`;
process.emitWarning = (() => {}) as typeof process.emitWarning;

import { restoreSandboxEnv } from "./restore-sandbox-env.js";

restoreSandboxEnv();

await import("./register-bedrock.js");
await import("../cli.js");
