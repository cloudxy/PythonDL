<template>
  <div class="space-y-6">
    <PageHeader title="爬虫采集" subtitle="管理数据采集任务">
      <template #actions>
        <Button variant="primary" @click="showCreateModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          新建任务
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '爬虫采集' }
    ]" />

    <Card>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <StatCard title="总任务数" :value="tasks.length.toString()" icon="📋" />
        <StatCard title="运行中" :value="runningCount.toString()" icon="⚙️" trend="实时" />
        <StatCard title="已完成" :value="completedCount.toString()" icon="✅" />
        <StatCard title="失败" :value="failedCount.toString()" icon="❌" trend="-2%" trend-negative />
      </div>

      <SearchBar
        v-model="searchQuery"
        placeholder="搜索任务名称、类型..."
        @search="handleSearch"
        @add="showCreateModal = true"
      />

      <Table
        :columns="columns"
        :data="filteredTasks"
        :loading="loading"
        :total="filteredTasks.length"
      >
        <template #cell-type="{ row }">
          <Badge :variant="getTypeBadgeVariant(row.type)">
            {{ typeMap[row.type] }}
          </Badge>
        </template>
        <template #cell-status="{ row }">
          <div class="flex items-center gap-2">
            <span :class="getStatusColor(row.status)" class="w-2 h-2 rounded-full"></span>
            <Badge :variant="getStatusBadgeVariant(row.status)">
              {{ statusMap[row.status] }}
            </Badge>
          </div>
        </template>
        <template #cell-progress="{ row }">
          <div class="flex items-center gap-2">
            <div class="flex-1 w-24 h-2 bg-secondary-200 rounded-full overflow-hidden">
              <div
                :class="getProgressColorClass(row.progress)"
                class="h-full rounded-full transition-all"
                :style="{ width: row.progress + '%' }"
              ></div>
            </div>
            <span class="text-sm text-secondary-600">{{ row.progress }}%</span>
          </div>
        </template>
        <template #cell-lastRun="{ row }">
          {{ formatDate(row.lastRun) }}
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button
              v-if="row.status === 'pending' || row.status === 'failed'"
              variant="primary"
              size="sm"
              @click="startTask(row)"
            >
              启动
            </Button>
            <Button
              v-else-if="row.status === 'running'"
              variant="warning"
              size="sm"
              @click="stopTask(row)"
            >
              停止
            </Button>
            <Button variant="ghost" size="sm" @click="viewLogs(row)">日志</Button>
            <Button variant="ghost" size="sm" @click="editTask(row)">编辑</Button>
            <Button variant="danger" size="sm" @click="deleteTask(row)">删除</Button>
          </div>
        </template>
      </Table>
    </Card>

    <!-- 新建/编辑任务模态框 -->
    <Modal
      v-model="showCreateModal"
      :title="editingTask ? '编辑任务' : '新建采集任务'"
      size="lg"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <FormSection title="任务配置">
        <Input
          v-model="taskForm.name"
          label="任务名称"
          placeholder="请输入任务名称"
          :error="formErrors.name"
          required
        />
        <Select
          v-model="taskForm.type"
          label="采集类型"
          :options="typeOptions"
          placeholder="请选择采集类型"
          :error="formErrors.type"
          required
        />
        <Input
          v-model="taskForm.url"
          label="目标 URL"
          placeholder="请输入目标网址"
          :error="formErrors.url"
          required
        />
        <Select
          v-model="taskForm.schedule"
          label="执行频率"
          :options="scheduleOptions"
          placeholder="请选择执行频率"
        />
        <Textarea
          v-model="taskForm.description"
          label="任务描述"
          placeholder="请输入任务描述"
          :rows="3"
        />
      </FormSection>
    </Modal>

    <!-- 日志模态框 -->
    <Modal
      v-model="showLogModal"
      title="任务日志"
      size="xl"
    >
      <div v-if="currentTask" class="space-y-4">
        <div class="flex items-center justify-between">
          <h4 class="font-semibold">{{ currentTask.name }}</h4>
          <Button variant="outline" size="sm" @click="exportLogs">
            <svg class="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            导出日志
          </Button>
        </div>
        <div class="bg-secondary-900 text-green-400 p-4 rounded-lg font-mono text-sm max-h-96 overflow-auto">
          <div v-for="(line, index) in logLines" :key="index" class="mb-1">
            {{ line }}
          </div>
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
import FormSection from '@/components/ui/FormSection.vue'
import SearchBar from '@/components/ui/SearchBar.vue'
import StatCard from '@/components/ui/StatCard.vue'

const loading = ref(false)
const submitting = ref(false)
const showCreateModal = ref(false)
const showLogModal = ref(false)
const searchQuery = ref('')
const editingTask = ref(null)

const typeMap = {
  stock: '股票数据',
  weather: '气象数据',
  fortune: '算命数据',
  consumption: '消费数据',
  general: '通用采集'
}

const statusMap = {
  pending: '待执行',
  running: '运行中',
  completed: '已完成',
  failed: '失败',
  stopped: '已停止'
}

const columns = [
  { key: 'name', label: '任务名称' },
  { key: 'type', label: '类型' },
  { key: 'status', label: '状态' },
  { key: 'progress', label: '进度' },
  { key: 'lastRun', label: '上次执行' },
  { key: 'nextRun', label: '下次执行' }
]

const tasks = ref([
  { id: 1, name: '股票行情采集', type: 'stock', status: 'running', progress: 65, url: 'https://example.com/stocks', schedule: 'hourly', lastRun: '2024-01-15 10:00:00', nextRun: '2024-01-15 11:00:00', description: '采集 A 股实时行情数据' },
  { id: 2, name: '气象数据采集', type: 'weather', status: 'completed', progress: 100, url: 'https://example.com/weather', schedule: 'daily', lastRun: '2024-01-15 06:00:00', nextRun: '2024-01-16 06:00:00', description: '采集全国气象数据' },
  { id: 3, name: '周易数据采集', type: 'fortune', status: 'pending', progress: 0, url: 'https://example.com/fortune', schedule: 'weekly', lastRun: '2024-01-08 00:00:00', nextRun: '2024-01-15 00:00:00', description: '采集周易相关数据' },
  { id: 4, name: '消费数据采集', type: 'consumption', status: 'failed', progress: 45, url: 'https://example.com/consumption', schedule: 'daily', lastRun: '2024-01-15 08:00:00', nextRun: '-', description: '采集消费统计数据' },
  { id: 5, name: '通用网页采集', type: 'general', status: 'stopped', progress: 30, url: 'https://example.com/general', schedule: 'manual', lastRun: '2024-01-14 15:00:00', nextRun: '-', description: '通用网页数据采集' }
])

const taskForm = reactive({
  name: '',
  type: 'general',
  url: '',
  schedule: 'manual',
  description: ''
})

const formErrors = reactive({
  name: '',
  type: '',
  url: ''
})

const typeOptions = [
  { value: 'stock', label: '股票数据' },
  { value: 'weather', label: '气象数据' },
  { value: 'fortune', label: '算命数据' },
  { value: 'consumption', label: '消费数据' },
  { value: 'general', label: '通用采集' }
]

const scheduleOptions = [
  { value: 'manual', label: '手动执行' },
  { value: 'hourly', label: '每小时' },
  { value: 'daily', label: '每天' },
  { value: 'weekly', label: '每周' },
  { value: 'monthly', label: '每月' }
]

const currentTask = ref(null)

const filteredTasks = computed(() => {
  if (!searchQuery.value) return tasks.value
  return tasks.value.filter(task =>
    task.name.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
    typeMap[task.type].includes(searchQuery.value)
  )
})

const runningCount = computed(() => tasks.value.filter(t => t.status === 'running').length)
const completedCount = computed(() => tasks.value.filter(t => t.status === 'completed').length)
const failedCount = computed(() => tasks.value.filter(t => t.status === 'failed').length)

const logLines = ref([
  '[2024-01-15 10:00:00] 任务启动...',
  '[2024-01-15 10:00:01] 连接目标服务器...',
  '[2024-01-15 10:00:02] 连接成功，开始采集...',
  '[2024-01-15 10:00:15] 已采集 100 条数据...',
  '[2024-01-15 10:00:30] 已采集 200 条数据...',
  '[2024-01-15 10:00:45] 数据采集完成，共 350 条',
  '[2024-01-15 10:00:46] 数据清洗中...',
  '[2024-01-15 10:00:50] 数据入库成功',
  '[2024-01-15 10:00:51] 任务完成'
])

const getTypeBadgeVariant = (type) => {
  const variants = {
    stock: 'primary',
    weather: 'info',
    fortune: 'warning',
    consumption: 'success',
    general: 'secondary'
  }
  return variants[type] || 'secondary'
}

const getStatusBadgeVariant = (status) => {
  const variants = {
    pending: 'secondary',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
    stopped: 'secondary'
  }
  return variants[status] || 'secondary'
}

const getStatusColor = (status) => {
  const colors = {
    pending: 'bg-secondary-400',
    running: 'bg-warning-500',
    completed: 'bg-success-500',
    failed: 'bg-danger-500',
    stopped: 'bg-secondary-500'
  }
  return colors[status] || 'bg-secondary-400'
}

const getProgressColorClass = (progress) => {
  if (progress < 50) return 'bg-warning-500'
  if (progress < 100) return 'bg-primary-500'
  return 'bg-success-500'
}

const formatDate = (date) => {
  if (!date || date === '-') return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const handleSearch = () => {}

const startTask = async (task) => {
  task.status = 'running'
  task.progress = 0
}

const stopTask = async (task) => {
  task.status = 'stopped'
}

const viewLogs = (task) => {
  currentTask.value = task
  showLogModal.value = true
}

const editTask = (task) => {
  editingTask.value = task
  Object.assign(taskForm, {
    name: task.name,
    type: task.type,
    url: task.url,
    schedule: task.schedule,
    description: task.description
  })
  showCreateModal.value = true
}

const deleteTask = async (task) => {
  if (confirm(`确定要删除任务 "${task.name}" 吗？`)) {
    tasks.value = tasks.value.filter(t => t.id !== task.id)
  }
}

const handleSubmit = async () => {
  submitting.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 1000))
    if (editingTask) {
      Object.assign(editingTask.value, { ...taskForm })
    } else {
      tasks.value.unshift({
        id: Date.now(),
        ...taskForm,
        status: 'pending',
        progress: 0,
        lastRun: '-',
        nextRun: '-'
      })
    }
    showCreateModal.value = false
    resetForm()
  } finally {
    submitting.value = false
  }
}

const resetForm = () => {
  editingTask.value = null
  Object.assign(taskForm, {
    name: '',
    type: 'general',
    url: '',
    schedule: 'manual',
    description: ''
  })
  Object.assign(formErrors, {
    name: '',
    type: '',
    url: ''
  })
}

const exportLogs = () => {
  alert('正在导出日志...')
}

onMounted(() => {
  loading.value = true
  setTimeout(() => {
    loading.value = false
  }, 500)
})
</script>
