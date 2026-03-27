<template>
  <div class="page">
    <header class="topbar">
      <div class="brand">
        <span class="title">Summaries</span>
        <span class="meta">
          {{ shownCount }} shown
          <span v-if="snapshotAt">· snapshot {{ snapshotAt }}</span>
        </span>
      </div>

      <div class="actions">
        <button class="btn" :disabled="loading" @click="refresh">Refresh</button>
        <span v-if="loading" class="hint">Loading…</span>
        <span v-if="error" class="error">{{ error }}</span>
      </div>
    </header>

    <div ref="scroller" class="scroller" @scroll.passive="onScroll">
      <div class="pull-hint" :class="{ show: pullHint.show }">
        <span>{{ pullHint.text }}</span>
      </div>

      <main class="grid">
        <article
          v-for="c in cards"
          :key="c.key"
          class="card"
          :style="{ backgroundColor: c.bg }"
        >
          <p class="text">{{ c.text }}</p>
        </article>
      </main>

      <div class="footer">
        <div v-if="loadingMore" class="hint">Loading more…</div>
        <div v-else-if="!hasMore && cards.length" class="hint">No more</div>
        <div v-else class="hint">Scroll down to load more · Scroll up to refresh</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'

type ClusterSummaryItem = {
  cluster_id: number
  summarize_result: string
  sort_time?: string
  url_set?: string[]
}
type ClusterSummaryResponse = {
  snapshot_at: string
  count: number
  has_more: boolean
  next_cursor_time?: string | null
  next_cursor_id?: number | null
  items: ClusterSummaryItem[]
}

const LIMIT = 20
const MAX_LEN = 500

// ⚠️ 网络保持最简单：直接请求你的 FastAPI。
// 如果你后面用 Vite proxy，把这里改成 '/clusters/recent?...' 即可。
const API_URL = 'http://127.0.0.1:8000/clusters/recent'

const scroller = ref<HTMLElement | null>(null)

const loading = ref(false)
const loadingMore = ref(false)
const error = ref('')

const snapshotAt = ref<string>('') // 翻页需要沿用
const cursorTime = ref<string | null>(null)
const cursorId = ref<number | null>(null)
const hasMore = ref(true)

const items = ref<ClusterSummaryItem[]>([])

const pullHint = reactive({
  show: false,
  text: 'Release to refresh',
})

function isBad(s: string) {
  const t = (s ?? '').trim()
  if (!t) return true
  if (t.length > MAX_LEN) return true
  if (/null/i.test(t)) return true
  return false
}

// 轻浅色调色板（浅色，且固定可控）
const pastel = [
  '#F7F2FF', // lavender milk
  '#F1F8FF', // sky milk
  '#FFF6F0', // peach milk
  '#F2FFF7', // mint milk
  '#FFF2F7', // rose milk
  '#F7FFF0', // lemon milk
  '#F3F4FF', // periwinkle milk
  '#F0FBFF', // ice milk
]

// 稳定“随机”颜色：cluster_id -> pastel index
function bgForId(id: number) {
  const x = (id * 2654435761) >>> 0
  return pastel[x % pastel.length]
}

// 首字母大写（对英文有效；中文不受影响）
function capitalizeFirst(s: string) {
  const t = (s ?? '').trim()
  if (!t) return t
  // 跳过前导非字母字符
  const m = t.match(/[A-Za-z]/)
  if (!m || m.index === undefined) return t
  const i = m.index
  return t.slice(0, i) + t.charAt(i).toUpperCase() + t.slice(i + 1)
}

const cards = computed(() => {
  return items.value.map((it) => ({
    key: String(it.cluster_id),
    text: capitalizeFirst(it.summarize_result),
    bg: bgForId(it.cluster_id),
  }))
})

const shownCount = computed(() => cards.value.length)

function buildUrl(params: Record<string, string | number | null | undefined>) {
  const u = new URL(API_URL, window.location.origin)
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined || v === '') continue
    u.searchParams.set(k, String(v))
  }
  return u.toString()
}

async function fetchPage(opts: { reset: boolean }) {
  if (loading.value || loadingMore.value) return

  if (opts.reset) loading.value = true
  else loadingMore.value = true

  error.value = ''
  try {
    const url = buildUrl({
      limit: LIMIT,
      snapshot_at: opts.reset ? null : snapshotAt.value || null,
      cursor_time: opts.reset ? null : cursorTime.value,
      cursor_id: opts.reset ? null : cursorId.value,
    })

    const resp = await fetch(url)
    if (!resp.ok) {
      const txt = await resp.text().catch(() => '')
      throw new Error(`HTTP ${resp.status}: ${txt || resp.statusText}`)
    }

    const data = (await resp.json()) as ClusterSummaryResponse

    if (opts.reset) {
      snapshotAt.value = data.snapshot_at
      items.value = (data.items || []).filter((x) => !isBad(x?.summarize_result))
    } else {
      const exist = new Set(items.value.map((x) => x.cluster_id))
      for (const it of data.items || []) {
        if (exist.has(it.cluster_id)) continue
        if (isBad(it?.summarize_result)) continue
        items.value.push(it)
      }
    }

    hasMore.value = !!data.has_more
    cursorTime.value = data.next_cursor_time ?? null
    cursorId.value = data.next_cursor_id ?? null
  } catch (e: any) {
    error.value = e?.message || String(e)
  } finally {
    loading.value = false
    loadingMore.value = false
  }
}

function refresh() {
  snapshotAt.value = ''
  cursorTime.value = null
  cursorId.value = null
  hasMore.value = true
  return fetchPage({ reset: true })
}

function loadMore() {
  if (!hasMore.value) return
  return fetchPage({ reset: false })
}

// scroll logic: down -> load more, up near top -> refresh
let lastScrollTop = 0
let ticking = false
let lastRefreshAt = 0
let lastLoadMoreAt = 0

function onScroll() {
  const el = scroller.value
  if (!el || ticking) return
  ticking = true

  requestAnimationFrame(() => {
    ticking = false
    const st = el.scrollTop
    const sh = el.scrollHeight
    const ch = el.clientHeight

    const now = Date.now()
    const goingUp = st < lastScrollTop
    const goingDown = st > lastScrollTop
    lastScrollTop = st

    // 顶部上拉刷新：靠近顶部 + 向上滚动
    if (goingUp && st <= 18 && now - lastRefreshAt > 1200 && !loading.value && !loadingMore.value) {
      lastRefreshAt = now
      pullHint.show = true
      pullHint.text = 'Refreshing…'
      refresh().finally(() => {
        setTimeout(() => (pullHint.show = false), 300)
      })
    }

    // 底部加载更多：靠近底部 + 向下滚动
    const nearBottom = st + ch >= sh - 80
    if (goingDown && nearBottom && hasMore.value && now - lastLoadMoreAt > 900 && !loading.value && !loadingMore.value) {
      lastLoadMoreAt = now
      loadMore()
    }
  })
}

onMounted(() => {
  refresh()
})
</script>

<style scoped>
/* 纯白背景 + 衬线字体 */
.page {
  height: 100vh;
  background: #ffffff;
  color: #111;
  font-family: ui-serif, Georgia, "Times New Roman", Times, serif;
  display: flex;
  flex-direction: column;
}

/* 顶栏（简洁、纯白系） */
.topbar {
  flex: 0 0 auto;
  padding: 14px 18px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.brand {
  display: flex;
  align-items: baseline;
  gap: 10px;
  min-width: 0;
}
.title {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.2px;
}
.meta {
  font-size: 12px;
  color: rgba(0, 0, 0, 0.55);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.actions {
  display: flex;
  align-items: center;
  gap: 10px;
}
.btn {
  border: 1px solid rgba(0, 0, 0, 0.10);
  background: #fff;
  border-radius: 10px;
  padding: 8px 10px;
  cursor: pointer;
  font-family: inherit;
}
.btn:hover {
  border-color: rgba(0, 0, 0, 0.18);
}
.btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
.hint {
  font-size: 12px;
  color: rgba(0, 0, 0, 0.55);
}
.error {
  font-size: 12px;
  color: #b00020;
}

/* 滚动容器 */
.scroller {
  position: relative;
  flex: 1 1 auto;
  overflow: auto;
  padding: 16px 18px 24px;
}

/* 顶部提示 */
.pull-hint {
  position: sticky;
  top: 0;
  z-index: 2;
  height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  transition: height 160ms ease;
}
.pull-hint.show {
  height: 28px;
}
.pull-hint span {
  font-size: 12px;
  color: rgba(0, 0, 0, 0.55);
}

/* 规则排列：网格 */
.grid {
  column-width: 260px;   /* 控制单列目标宽度：越小越窄 */
  column-gap: 24px;
}

@media (max-width: 560px) {
  .grid { column-width: 220px; }
}
/* 小圆角卡片 + 浅色随机背景 */
.card {
  break-inside: avoid;
  width: 100%;
  display: inline-block;
  margin: 0 0 24px;

  border-radius: 20px;
  padding: 20px;
  border: 1px solid rgba(0, 0, 0, 0.06);
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
}

/* 文本：首字母大写由 script 处理 */
.text {
  display: -webkit-box;
  -webkit-line-clamp: 30;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.footer {
  margin-top: 16px;
  display: flex;
  justify-content: center;
}
</style>

 