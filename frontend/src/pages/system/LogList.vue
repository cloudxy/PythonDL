<template>
  <div class="space-y-6">
    <PageHeader title="操作日志" subtitle="查看系统操作记录">
      <template #actions>
        <Button variant="outline" @click="handleExport">
          <svg class="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          导出日志
        </Button>
        <Button variant="danger" @click="handleClear">
          <svg class="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
          清理日志
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '系统管理' },
      { label: '操作日志' }
    ]" />

    <Card>
      <div class="flex flex-col sm:flex-row gap-4 mb-4">
        <SearchBar
          v-model="searchQuery"
          placeholder="搜索操作人、操作内容..."
          @search="handleSearch"
        />
        <div class="flex gap-2">
          <Select
            v-model="filterType"
            :options="typeOptions"
            placeholder="操作类型"
            class="w-32"
            @change="handleSearch"
          />
          <Select
            v-model="filterLevel"
            :options="levelOptions"
            placeholder="日志级别"
            class="w-32"
            @change="handleSearch"
          />
          <DatePicker
            v-model="dateRange"
            placeholder="日期范围"
            is-range
            @change="handleSearch"
          />
        </div>
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
        <template #cell-level="{ row }">
          <Badge :variant="getLevelBadgeVariant(row.level)">
            {{ levelMap[row.level] }}
          </Badge>
        </template>
        <template #cell-type="{ row }">
          <Badge :variant="getTypeBadgeVariant(row.type)">
            {{ typeMap[row.type] }}
          </Badge>
        </template>
        <template #cell-duration="{ row }">
          <span v-if="row.duration" class="text-secondary-600">{{ row.duration }}ms</span>
          <span v-else class="text-secondary-400">-</span>
        </template>
        <template #cell-createdAt="{ row }">
          {{ formatDate(row.createdAt) }}
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button variant="ghost" size="sm" @click="viewDetail(row)">详情</Button>
          </div>
        </template>
      </Table>
    </Card>

    <!-- 日志详情模态框 -->
    <Modal
      v-model="showDetailModal"
      title="日志详情"
      size="xl"
    >
      <div v-if="currentLog" class="space-y-4">
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="text-sm text-secondary-500">操作人</label>
            <p class="font-medium">{{ currentLog.username }}</p>
          </div>
          <div>
            <label class="text-sm text-secondary-500">操作类型</label>
            <Badge :variant="getTypeBadgeVariant(currentLog.type)">
              {{ typeMap[currentLog.type] }}
            </Badge>
          </div>
          <div>
            <label class="text-sm text-secondary-500">日志级别</label>
            <Badge :variant="getLevelBadgeVariant(currentLog.level)">
              {{ levelMap[currentLog.level] }}
            </Badge>
          </div>
          <div>
            <label class="text-sm text-secondary-500">耗时</label>
            <p class="font-medium">{{ currentLog.duration || '-' }}ms</p>
          </div>
          <div class="col-span-2">
            <label class="text-sm text-secondary-500">操作内容</label>
            <p class="font-medium">{{ currentLog.action }}</p>
          </div>
          <div class="col-span-2">
            <label class="text-sm text-secondary-500">请求方法</label>
            <Badge variant="secondary">{{ currentLog.method }}</Badge>
            <span class="ml-2 text-secondary-600">{{ currentLog.url }}</span>
          </div>
          <div class="col-span-2">
            <label class="text-sm text-secondary-500">IP 地址</label>
            <p class="font-medium">{{ currentLog.ip }}</p>
          </div>
          <div class="col-span-2">
            <label class="text-sm text-secondary-500">User Agent</label>
            <p class="text-sm text-secondary-600 font-mono">{{ currentLog.userAgent }}</p>
          </div>
          <div v-if="currentLog.requestData" class="col-span-2">
            <label class="text-sm text-secondary-500">请求数据</label>
            <pre class="mt-1 p-3 bg-secondary-50 rounded-lg text-sm overflow-auto max-h-40">{{ JSON.stringify(currentLog.requestData, null, 2) }}</pre>
          </div>
          <div v-if="currentLog.responseData" class="col-span-2">
            <label class="text-sm text-secondary-500">响应数据</label>
            <pre class="mt-1 p-3 bg-secondary-50 rounded-lg text-sm overflow-auto max-h-40">{{ JSON.stringify(currentLog.responseData, null, 2) }}</pre>
          </div>
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
import Select from '@/components/ui/Select.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import SearchBar from '@/components/ui/SearchBar.vue'
import DatePicker from '@/components/ui/DatePicker.vue'

const loading = ref(false)
const showDetailModal = ref(false)
const searchQuery = ref('')
const filterType = ref('')
const filterLevel = ref('')
const dateRange = ref([])
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const levelMap = {
  info: '信息',
  warning: '警告',
  error: '错误',
  debug: '调试'
}

const typeMap = {
  login: '登录',
  logout: '登出',
  create: '创建',
  update: '更新',
  delete: '删除',
  view: '查看',
  export: '导出',
  import: '导入',
  other: '其他'
}

const columns = [
  { key: 'id', label: 'ID', width: '80px' },
  { key: 'level', label: '级别' },
  { key: 'username', label: '操作人' },
  { key: 'type', label: '类型' },
  { key: 'action', label: '操作内容', width: '300px' },
  { key: 'ip', label: 'IP 地址' },
  { key: 'duration', label: '耗时' },
  { key: 'createdAt', label: '操作时间' }
]

const logs = ref([
  { id: 1, level: 'info', type: 'login', username: 'admin', action: '登录系统', ip: '192.168.1.100', duration: 120, method: 'POST', url: '/api/v1/auth/login', createdAt: '2024-01-15 10:30:00' },
  { id: 2, level: 'info', type: 'create', username: 'admin', action: '创建用户 test', ip: '192.168.1.100', duration: 250, method: 'POST', url: '/api/v1/admin/users', createdAt: '2024-01-15 10:35:00' },
  { id: 3, level: 'warning', type: 'update', username: 'user1', action: '更新配置信息', ip: '192.168.1.101', duration: 180, method: 'PUT', url: '/api/v1/admin/configs/1', createdAt: '2024-01-15 11:00:00' },
  { id: 4, level: 'error', type: 'delete', username: 'admin', action: '删除角色失败 - 权限不足', ip: '192.168.1.100', duration: 50, method: 'DELETE', url: '/api/v1/admin/roles/1', createdAt: '2024-01-15 11:15:00' },
  { id: 5, level: 'info', type: 'export', username: 'user2', action: '导出股票数据', ip: '192.168.1.102', duration: 3500, method: 'GET', url: '/api/v1/finance/stocks/export', createdAt: '2024-01-15 11:30:00' }
])

const typeOptions = [
  { value: '', label: '全部类型' },
  { value: 'login', label: '登录' },
  { value: 'logout', label: '登出' },
  { value: 'create', label: '创建' },
  { value: 'update', label: '更新' },
  { value: 'delete', label: '删除' },
  { value: 'view', label: '查看' },
  { value: 'export', label: '导出' },
  { value: 'import', label: '导入' },
  { value: 'other', label: '其他' }
]

const levelOptions = [
  { value: '', label: '全部级别' },
  { value: 'info', label: '信息' },
  { value: 'warning', label: '警告' },
  { value: 'error', label: '错误' },
  { value: 'debug', label: '调试' }
]

const currentLog = ref(null)

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const getLevelBadgeVariant = (level) => {
  const variants = {
    info: 'primary',
    warning: 'warning',
    error: 'danger',
    debug: 'secondary'
  }
  return variants[level] || 'secondary'
}

const getTypeBadgeVariant = (type) => {
  const variants = {
    login: 'success',
    logout: 'secondary',
    create: 'primary',
    update: 'warning',
    delete: 'danger',
    view: 'info',
    export: 'success',
    import: 'success',
    other: 'secondary'
  }
  return variants[type] || 'secondary'
}

const handleSearch = () => {
  page.value = 1
  loadLogs()
}

const handlePageChange = ({ page: newPage, pageSize: newSize }) => {
  page.value = newPage
  pageSize.value = newSize
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
  currentLog.value = {
    ...log,
    requestData: { username: 'admin' },
    responseData: { code: 200, message: 'success' },
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
  }
  showDetailModal.value = true
}

const handleExport = () => {
  alert('正在导出日志...')
}

const handleClear = () => {
  if (confirm('确定要清理 30 天前的日志吗？')) {
    alert('日志已清理')
  }
}

onMounted(() => {
  loadLogs()
})
</script>
