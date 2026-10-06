/**
 * NEMO/QW3 RC9 model-memory adapter reference.
 *
 * Invariants:
 * - QW3/KVMem is derived model memory, never an authority source.
 * - NEMO owns workspace identity, durable event history, admission and effects.
 * - selected-replay uses one-shot requests.
 * - query-replay profiles use explicit start/append/finish session operations.
 * - canonical session state is detachable host-side derived memory; executor
 *   slots are physically owned by the native executor pool, remain ephemeral compute
 *   affinity, and never become authority/durability truth.
 * - cold mounts, warm hits and scheduler backpressure are observable instead
 *   of being inferred from latency or session status.
 */
export interface NemoModelRequest {
  workspaceId: string;
  messages: Array<{ role: string; content: unknown }>;
  tools?: unknown[];
  maxTokens: number;
  signal?: AbortSignal;
}

export interface NemoSessionStepRequest extends NemoModelRequest {
  operation: "start" | "append" | "finish";
}

export interface Qw3MemoryReceipt {
  profile: string;
  state_coherence: "attention-kv" | "query-replay" | "selected-replay" | string;
  logical_prompt_tokens: number;
  selection_budget_tokens: number;
  generation_budget_tokens: number;
  requested_retrieval_method: string;
  semantic_expansion: string;
  query_conditioned: boolean;
  query_replay: boolean;
  strict_retrieval: boolean;
  retrieval_fallback_permitted: boolean;
  hybrid_state_exact_for_request: boolean;
}

export interface Qw3SessionReceipt {
  id: string;
  workspace_id: string | null;
  status: "hot" | "cold" | string;
  version: number;
  token_count: number;
  created_at: number;
  last_access_at: number;
  cold_rehydrates: number;
  host_bytes: number;
  hot: boolean;
  executor_slot: number | null;
}

export interface Qw3ExecutorSlotStatus {
  index: number;
  session_id: string | null;
  busy: boolean;
  dirty: boolean;
  generation: number;
  last_used_at: number;
  mounts: number;
  cold_mounts: number;
  fault_count: number;
  reset_count: number;
  active_lease_id: number;
}

export interface Qw3ExecutorSchedulerStatus {
  object: "kvmem.scheduler" | string;
  policy: "lru-affinity-fail-fast" | string;
  slot_count: number;
  busy_slots: number;
  mounted_sessions: number;
  lease_sequence: number;
  warm_hits: number;
  cold_mounts: number;
  backpressure_rejections: number;
  faulted_releases: number;
  forced_cold_resets: number;
  slots: Qw3ExecutorSlotStatus[];
}

export interface Qw3PhysicalExecutorRuntimeStatus {
  slot: number;
  runtime_id: string;
  installed: boolean;
  busy: boolean;
  dirty: boolean;
  mounted_session_id: string | null;
  runtime_generation: number;
  leases: number;
  successful_releases: number;
  faulted_releases: number;
  cold_resets: number;
  last_used_at: number;
  active_lease_id: number;
  lease_sequence: number;
  linked_scheduler_lease_id: number;
}

export interface Qw3PhysicalExecutorPoolStatus {
  object: "kvmem.executor_pool" | string;
  scope: "persistent-session-runtime" | string;
  runtime_abi: "executor-slot-runtime-v2" | string;
  configured_slots: number;
  certified_slots: number;
  installed_runtimes: number;
  runtimes: Qw3PhysicalExecutorRuntimeStatus[];
}

export interface Qw3ResourceAdmissionStatus {
  object: "kvmem.resources" | string;
  policy: "fail-fast-byte-envelope" | string;
  slot_vram_bytes: number;
  slot_host_bytes: number;
  slot_nvme_bytes: number;
  capacity_vram_bytes: number;
  capacity_host_bytes: number;
  capacity_nvme_bytes: number;
  used_vram_bytes: number;
  used_host_bytes: number;
  used_nvme_bytes: number;
  high_water_vram_bytes: number;
  high_water_host_bytes: number;
  high_water_nvme_bytes: number;
  max_inflight: number;
  inflight: number;
  high_water_inflight: number;
  admissions: number;
  rejections: number;
}

export interface Qw3SessionSnapshotReceipt {
  id: string;
  workspace_id: string | null;
  version: number;
  token_count: number;
  host_bytes: number;
  runtime_fingerprint: string;
  created_at: number;
  saved_at: number;
}

export interface NemoModelResult {
  raw: unknown;
  memoryReceipt?: Qw3MemoryReceipt;
  sessionReceipt?: Qw3SessionReceipt;
}

export interface Qw3KvmemStatus {
  object: "kvmem.status" | string;
  enabled: boolean;
  profile: string;
  state_coherence: string;
  hybrid_state_exact_for_supported_one_shot: boolean;
  kvmi_012_applies: boolean;
  strict_retrieval: boolean;
  method: string;
  retrieval_method: string;
  index_placement: string;
  update_mode: string;
  semantic_expansion: string;
  immutable_source_k: boolean;
  raw_k_nvme: boolean;
  configured_cpu_bytes: number;
  configured_nvme_bytes: number;
  gpu_memory_ratio: number;
  logical_context_tokens: number;
  selection_budget_tokens: number;
  prefill_budget_tokens: number;
  generation_budget_tokens: number;
  block_tokens: number;
  query_conditioned: boolean;
  query_replay: boolean;
  selected_replay_one_shot_only: boolean;
  serialized_generation: boolean;
  session_policy: "detachable-host-state" | string;
  session_state_abi_version: number;
  session_state_portability: "canonical-host-only" | string;
  session_scheduler: "lru-affinity-fail-fast" | string;
  executor_slots_configured: number;
  executor_slots_certified: number;
  physical_executor_runtime_abi: "executor-slot-runtime-v2" | string;
  physical_executor_runtimes_installed: number;
  scheduler_busy_slots: number;
  scheduler_warm_hits: number;
  scheduler_cold_mounts: number;
  scheduler_backpressure_rejections: number;
  scheduler_faulted_releases: number;
  scheduler_forced_cold_resets: number;
  resource_policy: "fail-fast-byte-envelope" | string;
  resource_slot_vram_bytes: number;
  resource_slot_host_bytes: number;
  resource_slot_nvme_bytes: number;
  resource_capacity_vram_bytes: number;
  resource_capacity_host_bytes: number;
  resource_capacity_nvme_bytes: number;
  resource_max_inflight: number;
  resource_inflight: number;
  resource_rejections: number;
  session_max: number;
  session_host_token_limit: number;
  session_host_byte_limit: number;
  registered_session_host_bytes: number;
  session_snapshot_enabled: boolean;
  registered_sessions: number;
  mounted_session_ids: string[];
  hot_session_id: string | null;
}

export interface Qw3ReadyStatus {
  status: "ready" | string;
  kvmem: Qw3KvmemStatus;
}

export interface WorkspaceBinding {
  workspaceId: string;
  qw3SessionId?: string;
  mode: "one-shot" | "persistent-session";
  state: "new" | "started" | "finished";
  createdAt: string;
  lastSessionVersion?: number;
}

export class Qw3KvmemRuntime {
  private status?: Qw3KvmemStatus;
  private readonly bindings = new Map<string, WorkspaceBinding>();

  constructor(
    private readonly baseUrl = "http://127.0.0.1:8080",
    private readonly requestTimeoutMs = 120_000,
  ) {}

  async preflight(): Promise<Qw3KvmemStatus> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/readyz`, {
      method: "GET",
    });
    if (!response.ok) throw new Error(`QW3 readiness failed: ${response.status}`);
    const ready = (await response.json()) as Qw3ReadyStatus;
    if (ready.status !== "ready" || !ready.kvmem?.enabled) {
      throw new Error("QW3 is reachable but KVMem is not ready");
    }
    if (ready.kvmem.executor_slots_configured > ready.kvmem.executor_slots_certified) {
      throw new Error(
        `QW3 configured ${ready.kvmem.executor_slots_configured} executor slots but only ` +
        `${ready.kvmem.executor_slots_certified} are certified by this runtime`,
      );
    }
    if (ready.kvmem.physical_executor_runtime_abi !== "executor-slot-runtime-v2") {
      throw new Error(
        `Unexpected QW3 physical executor ABI: ${ready.kvmem.physical_executor_runtime_abi}`,
      );
    }
    if (ready.kvmem.physical_executor_runtimes_installed < ready.kvmem.executor_slots_certified) {
      throw new Error(
        "QW3 reports fewer installed physical executor runtimes than certified slots",
      );
    }
    if (ready.kvmem.resource_policy !== "fail-fast-byte-envelope") {
      throw new Error(`Unexpected QW3 resource policy: ${ready.kvmem.resource_policy}`);
    }
    if (ready.kvmem.resource_max_inflight < 1) {
      throw new Error("QW3 resource admission reports no executable in-flight capacity");
    }
    this.status = ready.kvmem;
    return ready.kvmem;
  }

  async health(): Promise<Qw3KvmemStatus> {
    if (this.status) return this.status;
    return this.preflight();
  }

  bindWorkspace(workspaceId: string): WorkspaceBinding {
    const status = this.status;
    if (!status) throw new Error("preflight() must succeed before binding a workspace");
    const existing = this.bindings.get(workspaceId);
    if (existing) return existing;

    const oneShot = status.state_coherence === "selected-replay";
    const binding: WorkspaceBinding = {
      workspaceId,
      mode: oneShot ? "one-shot" : "persistent-session",
      qw3SessionId: oneShot ? undefined : this.sessionIdForWorkspace(workspaceId),
      state: "new",
      createdAt: new Date().toISOString(),
    };
    this.bindings.set(workspaceId, binding);
    return binding;
  }

  /** Full-transcript one-shot inference. Safe for selected-replay. */
  async infer(request: NemoModelRequest): Promise<NemoModelResult> {
    const status = await this.health();
    const binding = this.bindWorkspace(request.workspaceId);
    if (binding.mode !== "one-shot") {
      throw new Error(
        "This QW3 profile uses persistent sessions; call sessionStep(start|append|finish) with incremental fragments",
      );
    }
    return this.postCompletion(request, undefined, status);
  }

  /** Explicit incremental session operation. start/append must use maxTokens=0. */
  async sessionStep(request: NemoSessionStepRequest): Promise<NemoModelResult> {
    const status = await this.health();
    const binding = this.bindWorkspace(request.workspaceId);
    if (binding.mode !== "persistent-session" || !binding.qw3SessionId) {
      throw new Error("The active QW3 profile does not support persistent session mode");
    }
    if (request.operation !== "finish" && request.maxTokens !== 0) {
      throw new Error("KVMem session start/append requires maxTokens=0");
    }
    if (request.operation === "start" && binding.state !== "new") {
      // A deliberate start is a reset. Make that visible rather than silently
      // treating it as append.
      binding.state = "new";
    }
    if (request.operation !== "start" && binding.state === "new") {
      throw new Error("KVMem workspace session must be started before append/finish");
    }

    const result = await this.postCompletion(request, request.operation, status);
    if (!result.sessionReceipt) {
      throw new Error("QW3 persistent session response omitted kvmem_session receipt");
    }
    if (result.sessionReceipt.workspace_id !== request.workspaceId) {
      throw new Error("QW3 session/workspace binding mismatch");
    }
    binding.lastSessionVersion = result.sessionReceipt.version;
    binding.state = request.operation === "finish" ? "finished" : "started";
    return result;
  }

  async inspectScheduler(): Promise<Qw3ExecutorSchedulerStatus> {
    await this.health();
    const response = await this.fetchWithTimeout(`${this.baseUrl}/v1/kvmem/scheduler`, {
      method: "GET",
    });
    if (!response.ok) throw new Error(`QW3 scheduler inspect failed: ${response.status}`);
    return (await response.json()) as Qw3ExecutorSchedulerStatus;
  }

  async inspectExecutors(): Promise<Qw3PhysicalExecutorPoolStatus> {
    await this.health();
    const response = await this.fetchWithTimeout(`${this.baseUrl}/v1/kvmem/executors`, {
      method: "GET",
    });
    if (!response.ok) throw new Error(`QW3 executor pool inspect failed: ${response.status}`);
    const pool = (await response.json()) as Qw3PhysicalExecutorPoolStatus;
    if (pool.scope !== "persistent-session-runtime") {
      throw new Error(`Unexpected QW3 executor pool scope: ${pool.scope}`);
    }
    if (pool.runtime_abi !== "executor-slot-runtime-v2") {
      throw new Error(`Unexpected QW3 executor pool ABI: ${pool.runtime_abi}`);
    }
    return pool;
  }

  async inspectResources(): Promise<Qw3ResourceAdmissionStatus> {
    await this.health();
    const response = await this.fetchWithTimeout(`${this.baseUrl}/v1/kvmem/resources`, {
      method: "GET",
    });
    if (!response.ok) throw new Error(`QW3 resource inspect failed: ${response.status}`);
    return (await response.json()) as Qw3ResourceAdmissionStatus;
  }

  async inspectWorkspace(workspaceId: string): Promise<Qw3SessionReceipt | undefined> {
    const binding = this.bindings.get(workspaceId);
    if (!binding?.qw3SessionId) return undefined;
    const response = await this.fetchWithTimeout(
      `${this.baseUrl}/v1/kvmem/sessions/${encodeURIComponent(binding.qw3SessionId)}`,
      { method: "GET" },
    );
    if (response.status === 404) return undefined;
    if (!response.ok) throw new Error(`QW3 session inspect failed: ${response.status}`);
    const body = (await response.json()) as { kvmem_session?: Qw3SessionReceipt };
    return body.kvmem_session;
  }

  async snapshotWorkspace(workspaceId: string): Promise<Qw3SessionSnapshotReceipt> {
    const status = await this.health();
    if (!status.session_snapshot_enabled) {
      throw new Error("QW3 session snapshots are not configured");
    }
    const binding = this.bindings.get(workspaceId);
    if (!binding?.qw3SessionId) {
      throw new Error("Workspace has no persistent QW3 session to snapshot");
    }
    const response = await this.fetchWithTimeout(
      `${this.baseUrl}/v1/kvmem/sessions/${encodeURIComponent(binding.qw3SessionId)}/snapshot`,
      { method: "POST" },
    );
    if (!response.ok) throw new Error(`QW3 session snapshot failed: ${response.status}`);
    const body = (await response.json()) as { snapshot?: Qw3SessionSnapshotReceipt };
    if (!body.snapshot) throw new Error("QW3 snapshot response omitted snapshot receipt");
    if (body.snapshot.workspace_id !== workspaceId) {
      throw new Error("QW3 snapshot/workspace binding mismatch");
    }
    return body.snapshot;
  }

  async restoreWorkspace(workspaceId: string): Promise<Qw3SessionReceipt> {
    const status = await this.health();
    if (!status.session_snapshot_enabled) {
      throw new Error("QW3 session snapshots are not configured");
    }
    const binding = this.bindWorkspace(workspaceId);
    if (!binding.qw3SessionId) {
      throw new Error("Selected-replay one-shot workspaces do not use persistent session snapshots");
    }
    const response = await this.fetchWithTimeout(
      `${this.baseUrl}/v1/kvmem/sessions/${encodeURIComponent(binding.qw3SessionId)}/restore`,
      { method: "POST" },
    );
    if (!response.ok) throw new Error(`QW3 session restore failed: ${response.status}`);
    const body = (await response.json()) as { kvmem_session?: Qw3SessionReceipt };
    if (!body.kvmem_session) throw new Error("QW3 restore response omitted session receipt");
    if (body.kvmem_session.workspace_id !== workspaceId) {
      throw new Error("QW3 restored session/workspace binding mismatch");
    }
    binding.state = "started";
    binding.lastSessionVersion = body.kvmem_session.version;
    return body.kvmem_session;
  }

  async releaseWorkspace(workspaceId: string): Promise<void> {
    const binding = this.bindings.get(workspaceId);
    this.bindings.delete(workspaceId);
    if (!binding?.qw3SessionId) return;
    const response = await this.fetchWithTimeout(
      `${this.baseUrl}/v1/kvmem/sessions/${encodeURIComponent(binding.qw3SessionId)}`,
      { method: "DELETE" },
    );
    if (!response.ok && response.status !== 404) {
      throw new Error(`QW3 session release failed: ${response.status}`);
    }
  }

  private async postCompletion(
    request: NemoModelRequest,
    operation: NemoSessionStepRequest["operation"] | undefined,
    status: Qw3KvmemStatus,
  ): Promise<NemoModelResult> {
    const binding = this.bindWorkspace(request.workspaceId);
    const body: Record<string, unknown> = {
      model: "qw3",
      messages: request.messages,
      tools: request.tools,
      max_tokens: request.maxTokens,
    };
    if (operation) {
      body.kvmem_session_id = binding.qw3SessionId;
      body.kvmem_session_op = operation;
      body.kvmem_workspace_id = request.workspaceId;
    }

    const response = await this.fetchWithTimeout(
      `${this.baseUrl}/v1/chat/completions`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: request.signal,
      },
    );
    if (!response.ok) {
      throw new Error(`QW3 inference failed: ${response.status} ${await response.text()}`);
    }

    const raw = (await response.json()) as Record<string, unknown>;
    const receipt = raw.kvmem as Qw3MemoryReceipt | undefined;
    const sessionReceipt = raw.kvmem_session as Qw3SessionReceipt | undefined;
    if (status.strict_retrieval && receipt?.retrieval_fallback_permitted) {
      throw new Error("QW3 violated strict retrieval contract");
    }
    if (
      status.state_coherence === "selected-replay" &&
      receipt &&
      !receipt.hybrid_state_exact_for_request
    ) {
      throw new Error("QW3 selected-replay request did not report coherent state");
    }
    return { raw, memoryReceipt: receipt, sessionReceipt };
  }

  private sessionIdForWorkspace(workspaceId: string): string {
    const bytes = new TextEncoder().encode(workspaceId);
    let hash = 0xcbf29ce484222325n;
    const prime = 0x100000001b3n;
    const mask = 0xffffffffffffffffn;
    for (const byte of bytes) {
      hash ^= BigInt(byte);
      hash = (hash * prime) & mask;
    }
    return `nemo:${hash.toString(16).padStart(16, "0")}`;
  }

  private async fetchWithTimeout(input: string, init: RequestInit): Promise<Response> {
    if (init.signal) return fetch(input, init);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.requestTimeoutMs);
    try {
      return await fetch(input, { ...init, signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }
}
