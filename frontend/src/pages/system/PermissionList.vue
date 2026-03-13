<template>
  <div class="space-y-6">
    <PageHeader title="权限管理" subtitle="管理系统权限配置">
      <template #actions>
        <Button variant="primary" @click="showAddModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          新增权限
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '系统管理' },
      { label: '权限管理' }
    ]" />

    <Card>
      <div class="flex items-center justify-between mb-4">
        <SearchBar
          v-model="searchQuery"
          placeholder="搜索权限名称、编码..."
          @search="handleSearch"
          @add="showAddModal = true"
        />
        <Button variant="outline" @click="loadPermissions">
          <svg class="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          刷新
        </Button>
      </div>

      <div v-if="loading" class="flex justify-center py-12">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>

      <div v-else-if="permissions.length === 0" class="text-center py-12">
        <svg class="w-12 h-12 text-secondary-300 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <p class="text-secondary-500">暂无权限数据</p>
      </div>

      <div v-else class="space-y-4">
        <div v-for="module in groupedPermissions" :key="module.name" class="border border-secondary-200 rounded-lg">
          <div class="bg-secondary-50 px-4 py-3 border-b border-secondary-200 flex items-center justify-between">
            <h3 class="font-semibold text-secondary-900">{{ module.name }}</h3>
            <span class="text-sm text-secondary-500">{{ module.permissions.length }} 个权限</span>
          </div>
          <div class="divide-y divide-secondary-100">
            <div
              v-for="permission in module.permissions"
              :key="permission.id"
              class="px-4 py-3 flex items-center justify-between hover:bg-secondary-50"
            >
              <div class="flex items-center gap-3">
                <div class="w-8 h-8 bg-primary-100 rounded flex items-center justify-center">
                  <svg class="w-4 h-4 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
                  </svg>
                </div>
                <div>
                  <p class="font-medium text-secondary-900">{{ permission.name }}</p>
                  <p class="text-sm text-secondary-500">{{ permission.code }}</p>
                </div>
              </div>
              <div class="flex items-center gap-2">
                <Badge :variant="permission.type === 'read' ? 'primary' : 'warning'">
                  {{ permissionTypeMap[permission.type] }}
                </Badge>
                <Button variant="ghost" size="sm" @click="editPermission(permission)">编辑</Button>
                <Button variant="danger" size="sm" @click="deletePermission(permission)">删除</Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>

    <!-- 新增/编辑权限模态框 -->
    <Modal
      v-model="showAddModal"
      :title="editingPermission ? '编辑权限' : '新增权限'"
      size="lg"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <FormSection title="基本信息">
        <Input
          v-model="permissionForm.name"
          label="权限名称"
          placeholder="请输入权限名称"
          :error="formErrors.name"
          required
        />
        <Input
          v-model="permissionForm.code"
          label="权限编码"
          placeholder="请输入权限编码（如：user:view）"
          :error="formErrors.code"
          :disabled="!!editingPermission"
          required
        />
        <Select
          v-model="permissionForm.module"
          label="所属模块"
          :options="moduleOptions"
          placeholder="请选择所属模块"
          required
        />
        <Select
          v-model="permissionForm.type"
          label="权限类型"
          :options="typeOptions"
          placeholder="请选择权限类型"
          required
        />
        <Textarea
          v-model="permissionForm.description"
          label="权限描述"
          placeholder="请输入权限描述"
          :rows="3"
        />
      </FormSection>
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
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import SearchBar from '@/components/ui/SearchBar.vue'

const loading = ref(false)
const submitting = ref(false)
const showAddModal = ref(false)
const editingPermission = ref(null)
const searchQuery = ref('')

const permissionTypeMap = {
  read: '查看',
  create: '创建',
  update: '编辑',
  delete: '删除'
}

const columns = [
  { key: 'id', label: 'ID', width: '80px' },
  { key: 'name', label: '权限名称' },
  { key: 'code', label: '权限编码' },
  { key: 'module', label: '所属模块' },
  { key: 'type', label: '权限类型' },
  { key: 'description', label: '描述', width: '300px' }
]

const permissions = ref([
  { id: 1, name: '查看用户', code: 'user:view', module: 'system', type: 'read', description: '查看用户列表和详情' },
  { id: 2, name: '创建用户', code: 'user:create', module: 'system', type: 'create', description: '创建新用户' },
  { id: 3, name: '编辑用户', code: 'user:update', module: 'system', type: 'update', description: '编辑用户信息' },
  { id: 4, name: '删除用户', code: 'user:delete', module: 'system', type: 'delete', description: '删除用户' },
  { id: 5, name: '查看股票', code: 'stock:view', module: 'finance', type: 'read', description: '查看股票数据' },
  { id: 6, name: '股票预测', code: 'stock:predict', module: 'finance', type: 'create', description: '执行股票预测' },
  { id: 7, name: '查看气象数据', code: 'weather:view', module: 'weather', type: 'read', description: '查看气象数据' },
  { id: 8, name: '气象预测', code: 'weather:predict', module: 'weather', type: 'create', description: '执行气象预测' }
])

const permissionForm = reactive({
  name: '',
  code: '',
  module: '',
  type: 'read',
  description: ''
})

const formErrors = reactive({
  name: '',
  code: '',
  module: '',
  type: ''
})

const moduleOptions = [
  { value: 'system', label: '系统管理' },
  { value: 'finance', label: '金融分析' },
  { value: 'weather', label: '气象分析' },
  { value: 'fortune', label: '看相算命' },
  { value: 'consumption', label: '消费分析' },
  { value: 'crawler', label: '爬虫采集' }
]

const typeOptions = [
  { value: 'read', label: '查看' },
  { value: 'create', label: '创建' },
  { value: 'update', label: '编辑' },
  { value: 'delete', label: '删除' }
]

const groupedPermissions = computed(() => {
  const groups = {}
  permissions.value.forEach(permission => {
    if (!groups[permission.module]) {
      groups[permission.module] = {
        name: moduleOptions.find(m => m.value === permission.module)?.label || permission.module,
        permissions: []
      }
    }
    groups[permission.module].permissions.push(permission)
  })
  return Object.values(groups)
})

const handleSearch = () => {
  loadPermissions()
}

const loadPermissions = async () => {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
  } finally {
    loading.value = false
  }
}

const editPermission = (permission) => {
  editingPermission.value = permission
  Object.assign(permissionForm, {
    name: permission.name,
    code: permission.code,
    module: permission.module,
    type: permission.type,
    description: permission.description
  })
  showAddModal.value = true
}

const deletePermission = async (permission) => {
  if (confirm(`确定要删除权限 ${permission.name} 吗？`)) {
    permissions.value = permissions.value.filter(p => p.id !== permission.id)
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
  editingPermission.value = null
  Object.assign(permissionForm, {
    name: '',
    code: '',
    module: '',
    type: 'read',
    description: ''
  })
  Object.assign(formErrors, {
    name: '',
    code: '',
    module: '',
    type: ''
  })
}

onMounted(() => {
  loadPermissions()
})
</script>
