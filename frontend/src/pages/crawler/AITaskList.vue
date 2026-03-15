<template>
  <div class="space-y-6">
    <PageHeader title="AI 智能爬虫" subtitle="基于 LLM 的智能数据采集系统">
      <template #actions>
        <Button variant="primary" @click="showCreateModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          新建 AI 爬虫
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '首页', to: '/system/dashboard' },
      { label: 'AI 智能爬虫' }
    ]" />

    <!-- 统计卡片 -->
    <Card>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <StatCard title="总任务数" :value="tasks.length.toString()" icon="🤖" />
        <StatCard title="运行中" :value="runningCount.toString()" icon="⚡" trend="实时" />
        <StatCard title="已完成" :value="completedCount.toString()" icon="✅" />
        <StatCard title="提取数据" :value="totalExtracted.toString()" icon="📊" trend="+15%" />
      </div>

      <!-- 功能特性介绍 -->
      <div class="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <h3 class="font-semibold text-blue-900 mb-2">✨ AI 爬虫特性</h3>
        <ul class="text-sm text-blue-800 space-y-1">
          <li>🎯 <strong>自然语言配置</strong> - 只需描述需要的数据，AI 自动提取</li>
          <li>🧠 <strong>LLM 驱动</strong> - 支持 Ollama、OpenAI、Groq 等多种大模型</li>
          <li>⚡ <strong>智能识别</strong> - 自动识别网页结构，无需手动编写解析规则</li>
          <li>🔄 <strong>实时更新</strong> - 实时监控爬虫状态和数据提取进度</li>
        </ul>
      </div>

      <!-- 任务列表 -->
      <Table :columns="columns" :data="tasks" :loading="loading">
        <template #cell-type="{ row }">
          <Badge :variant="getTypeBadgeVariant(row.type)">
            {{ typeMap[row.type] }}
          </Badge>
        </template>
        <template #cell-status="{ row }">
          <div class="flex items-center gap-2">
            <span :class="getStatusColor(row.status)" class="w-2 h-2 rounded-full animate-pulse"></span>
            <Badge :variant="getStatusBadgeVariant(row.status)">
              {{ statusMap[row.status] }}
            </Badge>
          </div>
        </template>
        <template #cell-progress="{ row }">
          <div class="flex items-center gap-2">
            <div class="flex-1 w-32 h-2 bg-secondary-200 rounded-full overflow-hidden">
              <div
                :class="getProgressColorClass(row.progress)"
                class="h-full rounded-full transition-all"
                :style="{ width: row.progress + '%' }"
              ></div>
            </div>
            <span class="text-sm text-secondary-600">{{ row.progress }}%</span>
          </div>
        </template>
        <template #cell-llm="{ row }">
          <div class="text-sm">
            <div class="font-medium">{{ row.llm.provider }}</div>
            <div class="text-secondary-500">{{ row.llm.model }}</div>
          </div>
        </template>
        <template #cell-items="{ row }">
          <span class="font-medium">{{ row.items_extracted }}</span>
        </template>
        <template #cell-created_at="{ row }">
          {{ formatDateTime(row.created_at) }}
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button variant="outline" size="sm" @click="viewResult(row)">
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </Button>
            <Button variant="outline" size="sm" @click="refreshStatus(row)">
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </Button>
          </div>
        </template>
      </Table>
    </Card>

    <!-- 新建任务模态框 -->
    <Modal
      v-model="showCreateModal"
      :title="`新建 ${activeTab === 'smart' ? '智能单页' : '搜索'} 爬虫`"
      size="lg"
      max-height="75vh"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <Tabs v-model="activeTab" :tabs="tabOptions" />

      <div class="mt-4 space-y-4">
        <!-- 智能单页爬虫配置 -->
        <template v-if="activeTab === 'smart'">
          <Input
            v-model="smartForm.source"
            label="目标网址"
            placeholder="https://example.com"
            :error="formErrors.source"
            required
          />
        </template>

        <!-- 搜索爬虫配置 -->
        <template v-if="activeTab === 'search'">
          <Input
            v-model="searchForm.query"
            label="搜索关键词"
            placeholder="请输入搜索关键词"
            :error="formErrors.query"
            required
          />
          <Select
            v-model="searchForm.search_engine"
            label="搜索引擎"
            :options="searchEngineOptions"
          />
          <Input
            v-model.number="searchForm.max_results"
            label="最大结果数"
            type="number"
            :min="1"
            :max="50"
          />
        </template>

        <!-- 通用配置 -->
        <Textarea
          v-model="commonForm.prompt"
          label="提取要求"
          placeholder="请描述需要提取的数据，例如：'提取公司名称、创始人、联系方式、社交媒体链接'"
          :rows="4"
          :error="formErrors.prompt"
          required
        />

        <FormSection title="LLM 配置">
          <div class="grid grid-cols-2 gap-4">
            <Select
              v-model="commonForm.llm_provider"
              label="LLM 提供商"
              :options="llmProviderOptions"
            />
            <Input
              v-model="commonForm.llm_model"
              label="模型名称"
              placeholder="llama3.2"
              :error="formErrors.llm_model"
            />
          </div>
        </FormSection>

        <!-- 配置模板 -->
        <div class="p-3 bg-secondary-50 rounded-lg border border-secondary-200">
          <h4 class="font-medium text-sm mb-2">💡 配置模板</h4>
          <div class="space-y-2">
            <Button
              v-for="template in configTemplates"
              :key="template.name"
              variant="outline"
              size="sm"
              @click="applyTemplate(template)"
              class="text-xs"
            >
              {{ template.name }}
            </Button>
          </div>
        </div>
      </div>
    </Modal>

    <!-- 结果查看模态框 -->
    <Modal
      v-model="showResultModal"
      title="爬取结果"
      size="xl"
      max-height="80vh"
    >
      <div v-if="currentTask" class="space-y-4">
        <div class="flex items-center justify-between">
          <h4 class="font-semibold">{{ currentTask.name }}</h4>
          <Badge :variant="getStatusBadgeVariant(currentTask.status)">
            {{ statusMap[currentTask.status] }}
          </Badge>
        </div>

        <div class="grid grid-cols-3 gap-4 text-sm">
          <div>
            <div class="text-secondary-500">LLM</div>
            <div class="font-medium">{{ currentTask.llm?.provider }}/{{ currentTask.llm?.model }}</div>
          </div>
          <div>
            <div class="text-secondary-500">提取数据</div>
            <div class="font-medium">{{ currentTask.items_extracted }} 条</div>
          </div>
          <div>
            <div class="text-secondary-500">执行时间</div>
            <div class="font-medium">{{ currentTask.duration }}s</div>
          </div>
        </div>

        <!-- JSON 结果展示 -->
        <div class="border rounded-lg overflow-hidden">
          <div class="bg-secondary-50 px-4 py-2 border-b flex items-center justify-between">
            <span class="font-medium text-sm">提取结果</span>
            <Button variant="outline" size="sm" @click="copyResult">
              <svg class="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              复制
            </Button>
          </div>
          <pre class="bg-secondary-900 text-green-400 p-4 text-sm max-h-96 overflow-auto"><code>{{ JSON.stringify(currentTask.result, null, 2) }}</code></pre>
        </div>
      </div>
    </Modal>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Input from '@/components/ui/Input.vue'
import Textarea from '@/components/ui/Textarea.vue'
import Select from '@/components/ui/Select.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import Tabs from '@/components/ui/Tabs.vue'
import FormSection from '@/components/ui/FormSection.vue'
import StatCard from '@/components/ui/StatCard.vue'
import api from '@/api'

const loading = ref(false)
const submitting = ref(false)
const showCreateModal = ref(false)
const showResultModal = ref(false)
const activeTab = ref('smart')

const typeMap = {
  smart: '智能单页',
  search: '搜索引擎'
}

const statusMap = {
  pending: '待执行',
  running: '运行中',
  completed: '已完成',
  failed: '失败'
}

const columns = [
  { key: 'name', label: '任务名称' },
  { key: 'type', label: '类型' },
  { key: 'status', label: '状态' },
  { key: 'progress', label: '进度' },
  { key: 'llm', label: 'LLM' },
  { key: 'items_extracted', label: '提取数据' },
  { key: 'created_at', label: '创建时间' }
]

const tasks = ref([])

const smartForm = reactive({
  source: '',
  prompt: '',
  llm_provider: 'ollama',
  llm_model: 'llama3.2'
})

const searchForm = reactive({
  query: '',
  search_engine: 'google',
  max_results: 10,
  prompt: '',
  llm_provider: 'ollama',
  llm_model: 'llama3.2'
})

const commonForm = reactive({
  prompt: '',
  llm_provider: 'ollama',
  llm_model: 'llama3.2'
})

const formErrors = reactive({
  source: '',
  query: '',
  prompt: '',
  llm_model: ''
})

const tabOptions = [
  { value: 'smart', label: '智能单页爬虫' },
  { value: 'search', label: '搜索引擎爬虫' }
]

const searchEngineOptions = [
  { value: 'google', label: 'Google' },
  { value: 'baidu', label: '百度' },
  { value: 'bing', label: 'Bing' }
]

const llmProviderOptions = [
  { value: 'ollama', label: 'Ollama (本地)' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'groq', label: 'Groq' }
]

const configTemplates = [
  {
    name: '📊 公司信息',
    prompt: '提取公司名称、创始人、联系方式、社交媒体链接、公司地址'
  },
  {
    name: '📰 新闻文章',
    prompt: '提取文章标题、作者、发布时间、正文内容、标签分类'
  },
  {
    name: '🛍️ 商品信息',
    prompt: '提取商品名称、价格、描述、规格、用户评价、库存状态'
  },
  {
    name: '👤 人物资料',
    prompt: '提取人物姓名、职位、简介、照片、教育背景、工作经历'
  }
]

const currentTask = ref(null)

const runningCount = computed(() => tasks.value.filter(t => t.status === 'running').length)
const completedCount = computed(() => tasks.value.filter(t => t.status === 'completed').length)
const totalExtracted = computed(() => tasks.value.reduce((sum, t) => sum + (t.items_extracted || 0), 0))

const getTypeBadgeVariant = (type) => {
  return type === 'smart' ? 'primary' : 'info'
}

const getStatusBadgeVariant = (status) => {
  const variants = {
    pending: 'secondary',
    running: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return variants[status] || 'secondary'
}

const getStatusColor = (status) => {
  const colors = {
    pending: 'bg-secondary-400',
    running: 'bg-warning-500',
    completed: 'bg-success-500',
    failed: 'bg-danger-500'
  }
  return colors[status] || 'bg-secondary-400'
}

const getProgressColorClass = (progress) => {
  if (progress < 50) return 'bg-warning-500'
  if (progress < 100) return 'bg-primary-500'
  return 'bg-success-500'
}

const formatDateTime = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const loadTasks = async () => {
  loading.value = true
  try {
    // TODO: 调用 API 获取任务列表
    // 模拟数据
    tasks.value = [
      {
        id: 1,
        name: '智能单页爬虫',
        type: 'smart',
        status: 'completed',
        progress: 100,
        llm: { provider: 'ollama', model: 'llama3.2' },
        items_extracted: 15,
        created_at: '2024-01-15 10:00:00',
        duration: 3.5,
        result: { company: 'Example Inc', founder: 'John Doe' }
      }
    ]
  } finally {
    loading.value = false
  }
}

const handleSubmit = async () => {
  submitting.value = true
  try {
    if (activeTab.value === 'smart') {
      // 创建智能单页爬虫任务
      const response = await api.post('/ai-crawler/ai/smart-scraper', {
        prompt: commonForm.prompt,
        source: smartForm.source,
        llm_provider: commonForm.llm_provider,
        llm_model: commonForm.llm_model
      })
      
      if (response.success) {
        tasks.value.unshift({
          id: Date.now(),
          name: `SmartScraper: ${smartForm.source}`,
          type: 'smart',
          status: 'pending',
          progress: 0,
          llm: { provider: commonForm.llm_provider, model: commonForm.llm_model },
          items_extracted: 0,
          created_at: new Date().toISOString(),
          result: null
        })
        showCreateModal.value = false
        resetForm()
      }
    } else {
      // 创建搜索爬虫任务
      const response = await api.post('/ai-crawler/ai/search-graph', {
        prompt: commonForm.prompt,
        query: searchForm.query,
        search_engine: searchForm.search_engine,
        max_results: searchForm.max_results,
        llm_provider: commonForm.llm_provider,
        llm_model: commonForm.llm_model
      })
      
      if (response.success) {
        tasks.value.unshift({
          id: Date.now(),
          name: `SearchGraph: ${searchForm.query}`,
          type: 'search',
          status: 'pending',
          progress: 0,
          llm: { provider: commonForm.llm_provider, model: commonForm.llm_model },
          items_extracted: 0,
          created_at: new Date().toISOString(),
          result: null
        })
        showCreateModal.value = false
        resetForm()
      }
    }
  } catch (error) {
    console.error('创建任务失败:', error)
  } finally {
    submitting.value = false
  }
}

const resetForm = () => {
  smartForm.source = ''
  searchForm.query = ''
  searchForm.search_engine = 'google'
  searchForm.max_results = 10
  commonForm.prompt = ''
  commonForm.llm_provider = 'ollama'
  commonForm.llm_model = 'llama3.2'
  Object.assign(formErrors, {
    source: '',
    query: '',
    prompt: '',
    llm_model: ''
  })
}

const viewResult = async (task) => {
  currentTask.value = task
  showResultModal.value = true
  
  // 如果是运行中的任务，刷新状态
  if (task.status === 'running') {
    await refreshStatus(task)
  }
}

const refreshStatus = async (task) => {
  try {
    const response = await api.get(`/ai-crawler/ai/task/${task.id}/status`)
    if (response.success) {
      Object.assign(task, response.data)
    }
  } catch (error) {
    console.error('刷新状态失败:', error)
  }
}

const copyResult = () => {
  if (currentTask.value?.result) {
    navigator.clipboard.writeText(JSON.stringify(currentTask.value.result, null, 2))
    alert('结果已复制到剪贴板')
  }
}

const applyTemplate = (template) => {
  commonForm.prompt = template.prompt
}

onMounted(() => {
  loadTasks()
  
  // 定时刷新运行中的任务
  setInterval(() => {
    tasks.value
      .filter(t => t.status === 'running')
      .forEach(task => refreshStatus(task))
  }, 5000)
})
</script>
