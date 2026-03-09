<template>
  <div class="space-y-6">
    <PageHeader title="操作日志" subtitle="查看系统操作记录" />

    <Breadcrumb :items="[
      { label: '系统管理' },
      { label: '操作日志' }
    ]" />

    <Card>
      <!-- 筛选条件 -->
      <div class="flex flex-wrap items-center gap-4 mb-6">
        <div class="flex-1 min-w-[200px]">
          <Input v-model="filters.keyword" placeholder="搜索操作内容..." />
        </div>
        <Select
          v-model="filters.module"
          :options="moduleOptions"
          placeholder="选择模块"
        />
        <Select
          v-model="filters.action"
          :options="actionOptions"
          placeholder="选择操作"
        />
        <DatePicker v-model="filters.startDate" label="" placeholder="开始日期" />
        <DatePicker v-model="filters.endDate" label="" placeholder="结束日期" />
        <Button variant="primary" @click="handleSearch">搜索</Button>
        <Button variant="secondary" @click="resetFilters">重置</Button>
        <Button variant="secondary" @click="exportLogs">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          导出
        </Button>
      </div>

      <Table
        :columns="columns"
        :data="logs"
        :loading="loading"
        :total="total"
        :page="page"
        :page-size="pageSize"
        @page-change="handlePageChange"
      >
        <template #cell-module="{ row }">
          <Badge :variant="getModuleVariant(row.module)">{{ row.module }}</Badge>
        </template>
        <template #cell-action="{ row }">
          <Badge :variant="getActionVariant(row.action)">{{ row.action }}</Badge>
        </template>
        <template #cell-status="{ row }">
          <StatusIndicator :status="row.status === 'success' ? 'success' : 'danger'" :text="row.status === 'success' ? '成功' : '失败'" />
        </template>
        <template #cell-createdAt="{ row }">
          {{ formatDate(row.createdAt) }}
        </template>
        <template #actions="{ row }">
          <Button variant="ghost" size="sm" @click="viewDetail(row)">详情</Button>
        </template>
      </Table>
    </Card>

    <!-- 详情模态框 -->
    <Modal
      v-model="showDetailModal"
      title="日志详情"
      size="lg"
    >
      <div v-if="currentLog" class="space-y-4">
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="text-sm text-secondary-500">操作用户</label>
            <p class="font-medium">{{ currentLog.username }}</p>
          </div>
          <div>
            <label class="text-sm text-secondary-500">操作模块</label>
            <p class="font-medium">{{ currentLog.module }}</p>
          </div>
          <div>
            <label class="text-sm text-secondary-500">操作类型</label>
            <p class="font-medium">{{ currentLog.action }}</p>
          </div>
          <div>
            <label class="text-sm text-secondary-500">操作状态</label>
            <StatusIndicator :status="currentLog.status === 'success' ? 'success' : 'danger'" :text="currentLog.status === 'success' ? '成功' : '失败'" />
          </div>
          <div>
            <label class="text-sm text-secondary-500">IP地址</label>
            <p class="font-medium">{{ currentLog.ip }}</p>
          </div>
          <div>
            <label class="text-sm text-secondary-500">操作时间</label>
            <p class="font-medium">{{ formatDate(currentLog.createdAt) }}</p>
          </div>
        </div>
        <div>
          <label class="text-sm text-secondary-500">操作内容</label>
          <p class="font-medium">{{ currentLog.content }}</p>
        </div>
        <div>
          <label class="text-sm text-secondary-500">请求参数</label>
          <pre class="mt-1 p-3 bg-secondary-50 rounded-lg text-sm overflow-auto">{{ currentLog.params }}</pre>
        </div>
        <div>
          <label class="text-sm text-secondary-500">响应结果</label>
          <pre class="mt-1 p-3 bg-secondary-50 rounded-lg text-sm overflow-auto">{{ currentLog.response }}</pre>
        </div>
        <div>
          <label class="text-sm text-secondary-500">User-Agent</label>
          <p class="text-sm text-secondary-600 break-all">{{ currentLog.userAgent }}</p>
        </div>
      </div>
    </Modal>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Input from '@/components/ui/Input.vue'
import Select from '@/components/ui/Select.vue'
import DatePicker from '@/components/ui/DatePicker.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import StatusIndicator from '@/components/ui/StatusIndicator.vue'

const loading = ref(false)
const showDetailModal = ref(false)
const currentLog = ref(null)
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const filters = reactive({
  keyword: '',
  module: '',
  action: '',
  startDate: '',
  endDate: ''
})

const columns = [
  { key: 'id', label: 'ID', width: '80px' },
  { key: 'username', label: '操作用户' },
  { key: 'module', label: '操作模块' },
  { key: 'action', label: '操作类型' },
  { key: 'content', label: '操作内容' },
  { key: 'ip', label: 'IP地址' },
  { key: 'status', label: '状态' },
  { key: 'createdAt', label: '操作时间' }
]

const logs = ref([
  { id: 1, username: 'admin', module: '用户管理', action: '登录', content: '用户登录系统', ip: '192.168.1.100', status: 'success', createdAt: '2024-01-15 10:30:00', params: '{}', response: '{}', userAgent: 'Mozilla/5.0...' },
  { id: 2, username: 'admin', module: '用户管理', action: '新增', content: '新增用户：user1', ip: '192.168.1.100', status: 'success', createdAt: '2024-01-15 10:35:00', params: '{}', response: '{}', userAgent: 'Mozilla/5.0...' },
  { id: 3, username: 'admin', module: '系统配置', action: '修改', content: '修改系统配置', ip: '192.168.1.100', status: 'success', createdAt: '2024-01-15 11:00:00', params: '{}', response: '{}', userAgent: 'Mozilla/5.0...' },
  { id: 4, username: 'user1', module: '金融分析', action: '查询', content: '查询股票数据', ip: '192.168.1.101', status: 'success', createdAt: '2024-01-15 14:20:00', params: '{}', response: '{}', userAgent: 'Mozilla/5.0...' },
  { id: 5, username: 'user1', module: '用户管理', action: '登录', content: '用户登录系统', ip: '192.168.1.101', status: 'failed', createdAt: '2024-01-15 14:15:00', params: '{}', response: '{}', userAgent: 'Mozilla/5.0...' }
])

const moduleOptions = [
  { value: 'user', label: '用户管理' },
  { value: 'role', label: '角色管理' },
  { value: 'config', label: '系统配置' },
  { value: 'finance', label: '金融分析' },
  { value: 'weather', label: '气象分析' }
]

const actionOptions = [
  { value: 'login', label: '登录' },
  { value: 'create', label: '新增' },
  { value: 'update', label: '修改' },
  { value: 'delete', label: '删除' },
  { value: 'query', label: '查询' }
]

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const getModuleVariant = (module) => {
  const variants = {
    '用户管理': 'primary',
    '角色管理': 'success',
    '系统配置': 'warning',
    '金融分析': 'danger'
  }
  return variants[module] || 'secondary'
}

const getActionVariant = (action) => {
  const variants = {
    '登录': 'primary',
    '新增': 'success',
    '修改': 'warning',
    '删除': 'danger',
    '查询': 'secondary'
  }
  return variants[action] || 'secondary'
}

const handleSearch = () => {
  page.value = 1
  loadLogs()
}

const resetFilters = () => {
  Object.assign(filters, {
    keyword: '',
    module: '',
    action: '',
    startDate: '',
    endDate: ''
  })
  handleSearch()
}

const handlePageChange = ({ page: newPage }) => {
  page.value = newPage
  loadLogs()
}

const loadLogs = async () => {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
    total.value = logs.value.length
  } finally {
    loading.value = false
  }
}

const viewDetail = (log) => {
  currentLog.value = log
  showDetailModal.value = true
}

const exportLogs = () => {
  alert('日志导出中...')
}

onMounted(() => {
  loadLogs()
})
</script>
