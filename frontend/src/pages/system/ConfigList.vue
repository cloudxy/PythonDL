<template>
  <div class="space-y-6">
    <PageHeader title="系统配置" subtitle="管理系统运行参数">
      <template #actions>
        <Button variant="primary" @click="showAddModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          新增配置
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '系统管理' },
      { label: '系统配置' }
    ]" />

    <Card>
      <SearchBar
        v-model="searchQuery"
        placeholder="搜索配置名称、键名..."
        @search="handleSearch"
        @add="showAddModal = true"
      />

      <Table
        :columns="columns"
        :data="configs"
        :loading="loading"
        :total="total"
        :page="page"
        :page-size="pageSize"
        @page-change="handlePageChange"
      >
        <template #cell-type="{ row }">
          <Badge :variant="getTypeBadgeVariant(row.type)">
            {{ typeMap[row.type] }}
          </Badge>
        </template>
        <template #cell-status="{ row }">
          <Badge :variant="row.status === 'active' ? 'success' : 'secondary'">
            {{ row.status === 'active' ? '启用' : '禁用' }}
          </Badge>
        </template>
        <template #cell-value="{ row }">
          <span class="text-secondary-600">{{ displayValue(row) }}</span>
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button variant="ghost" size="sm" @click="editConfig(row)">编辑</Button>
            <Button variant="danger" size="sm" @click="deleteConfig(row)">删除</Button>
          </div>
        </template>
      </Table>
    </Card>

    <!-- 新增/编辑配置模态框 -->
    <Modal
      v-model="showAddModal"
      :title="editingConfig ? '编辑配置' : '新增配置'"
      size="lg"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <FormSection title="基本信息">
        <Input
          v-model="configForm.name"
          label="配置名称"
          placeholder="请输入配置名称"
          :error="formErrors.name"
          required
        />
        <Input
          v-model="configForm.key"
          label="配置键名"
          placeholder="请输入配置键名（如：site.name）"
          :error="formErrors.key"
          :disabled="!!editingConfig"
          required
        />
        <Textarea
          v-model="configForm.value"
          label="配置值"
          placeholder="请输入配置值"
          :rows="3"
          :error="formErrors.value"
          required
        />
        <Select
          v-model="configForm.type"
          label="配置类型"
          :options="typeOptions"
          placeholder="请选择配置类型"
          required
        />
        <Textarea
          v-model="configForm.description"
          label="配置描述"
          placeholder="请输入配置描述"
          :rows="2"
        />
        <Select
          v-model="configForm.status"
          label="状态"
          :options="statusOptions"
          placeholder="请选择状态"
        />
      </FormSection>
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
import Textarea from '@/components/ui/Textarea.vue'
import Select from '@/components/ui/Select.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import SearchBar from '@/components/ui/SearchBar.vue'

const loading = ref(false)
const submitting = ref(false)
const showAddModal = ref(false)
const editingConfig = ref(null)
const searchQuery = ref('')
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const typeMap = {
  string: '文本',
  number: '数字',
  boolean: '布尔',
  json: 'JSON',
  file: '文件'
}

const columns = [
  { key: 'id', label: 'ID', width: '80px' },
  { key: 'name', label: '配置名称' },
  { key: 'key', label: '配置键名' },
  { key: 'value', label: '配置值', width: '300px' },
  { key: 'type', label: '类型' },
  { key: 'description', label: '描述', width: '250px' },
  { key: 'status', label: '状态' }
]

const configs = ref([
  { id: 1, name: '网站名称', key: 'site.name', value: 'PythonDL 智能分析平台', type: 'string', description: '网站标题', status: 'active' },
  { id: 2, name: '网站 Logo', key: 'site.logo', value: '/logo.png', type: 'file', description: '网站 Logo 地址', status: 'active' },
  { id: 3, name: '最大上传文件大小', key: 'upload.max_size', value: '10', type: 'number', description: '单位：MB', status: 'active' },
  { id: 4, name: '启用验证码', key: 'captcha.enabled', value: 'true', type: 'boolean', description: '登录时是否启用验证码', status: 'active' },
  { id: 5, name: 'API 限流', key: 'api.rate_limit', value: '100', type: 'number', description: '每分钟请求数限制', status: 'active' }
])

const configForm = reactive({
  name: '',
  key: '',
  value: '',
  type: 'string',
  description: '',
  status: 'active'
})

const formErrors = reactive({
  name: '',
  key: '',
  value: ''
})

const typeOptions = [
  { value: 'string', label: '文本' },
  { value: 'number', label: '数字' },
  { value: 'boolean', label: '布尔' },
  { value: 'json', label: 'JSON' },
  { value: 'file', label: '文件' }
]

const statusOptions = [
  { value: 'active', label: '启用' },
  { value: 'inactive', label: '禁用' }
]

const displayValue = (row) => {
  if (row.type === 'file') {
    return '📁 文件'
  }
  if (row.type === 'json') {
    return '📄 JSON'
  }
  if (row.value && row.value.length > 50) {
    return row.value.substring(0, 50) + '...'
  }
  return row.value
}

const getTypeBadgeVariant = (type) => {
  const variants = {
    string: 'primary',
    number: 'success',
    boolean: 'warning',
    json: 'info',
    file: 'secondary'
  }
  return variants[type] || 'secondary'
}

const handleSearch = () => {
  page.value = 1
  loadConfigs()
}

const handlePageChange = ({ page: newPage, pageSize: newSize }) => {
  page.value = newPage
  pageSize.value = newSize
  loadConfigs()
}

const loadConfigs = async () => {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
    total.value = configs.value.length
  } finally {
    loading.value = false
  }
}

const editConfig = (config) => {
  editingConfig.value = config
  Object.assign(configForm, {
    name: config.name,
    key: config.key,
    value: config.value,
    type: config.type,
    description: config.description,
    status: config.status
  })
  showAddModal.value = true
}

const deleteConfig = async (config) => {
  if (confirm(`确定要删除配置 ${config.name} 吗？`)) {
    configs.value = configs.value.filter(c => c.id !== config.id)
  }
}

const handleSubmit = async () => {
  submitting.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 1000))
    showAddModal.value = false
    resetForm()
  } finally {
    submitting.value = false
  }
}

const resetForm = () => {
  editingConfig.value = null
  Object.assign(configForm, {
    name: '',
    key: '',
    value: '',
    type: 'string',
    description: '',
    status: 'active'
  })
  Object.assign(formErrors, {
    name: '',
    key: '',
    value: ''
  })
}

onMounted(() => {
  loadConfigs()
})
</script>
