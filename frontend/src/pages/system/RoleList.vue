<template>
  <div class="space-y-6">
    <PageHeader title="角色管理" subtitle="管理系统角色和权限分配">
      <template #actions>
        <Button variant="primary" @click="showAddModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          新增角色
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '系统管理' },
      { label: '角色管理' }
    ]" />

    <Card>
      <SearchBar
        v-model="searchQuery"
        placeholder="搜索角色名称、编码..."
        @search="handleSearch"
        @add="showAddModal = true"
      />

      <Table
        :columns="columns"
        :data="roles"
        :loading="loading"
        :total="total"
        :page="page"
        :page-size="pageSize"
        @page-change="handlePageChange"
      >
        <template #cell-status="{ row }">
          <Badge :variant="row.status === 'active' ? 'success' : 'danger'">
            {{ row.status === 'active' ? '启用' : '禁用' }}
          </Badge>
        </template>
        <template #cell-permissions="{ row }">
          <Badge variant="secondary">{{ row.permissionCount }} 个权限</Badge>
        </template>
        <template #cell-createdAt="{ row }">
          {{ formatDate(row.createdAt) }}
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button variant="ghost" size="sm" @click="viewPermissions(row)">权限</Button>
            <Button variant="ghost" size="sm" @click="editRole(row)">编辑</Button>
            <Button variant="danger" size="sm" @click="deleteRole(row)">删除</Button>
          </div>
        </template>
      </Table>
    </Card>

    <!-- 新增/编辑角色模态框 -->
    <Modal
      v-model="showAddModal"
      :title="editingRole ? '编辑角色' : '新增角色'"
      size="lg"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <FormSection title="基本信息">
        <Input
          v-model="roleForm.name"
          label="角色名称"
          placeholder="请输入角色名称"
          :error="formErrors.name"
          required
        />
        <Input
          v-model="roleForm.code"
          label="角色编码"
          placeholder="请输入角色编码（如：admin）"
          :error="formErrors.code"
          :disabled="!!editingRole"
          required
        />
        <Textarea
          v-model="roleForm.description"
          label="角色描述"
          placeholder="请输入角色描述"
          :rows="3"
        />
        <Select
          v-model="roleForm.status"
          label="状态"
          :options="statusOptions"
          placeholder="请选择状态"
        />
      </FormSection>
    </Modal>

    <!-- 权限配置模态框 -->
    <Modal
      v-model="showPermissionModal"
      title="配置权限"
      size="xl"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handlePermissionSubmit"
    >
      <div class="space-y-4">
        <div v-for="module in permissionModules" :key="module.name" class="border border-secondary-200 rounded-lg p-4">
          <div class="flex items-center gap-2 mb-3">
            <Checkbox
              v-model="module.checked"
              :label="module.name"
              @change="toggleModulePermissions(module)"
            />
          </div>
          <div class="grid grid-cols-2 md:grid-cols-3 gap-2 ml-6">
            <div
              v-for="permission in module.permissions"
              :key="permission.id"
              class="flex items-center gap-2"
            >
              <Checkbox
                v-model="permission.checked"
                :label="permission.name"
                :disabled="!module.checked"
              />
            </div>
          </div>
        </div>
      </div>
    </Modal>

    <!-- 角色详情模态框 -->
    <Modal
      v-model="showDetailModal"
      title="角色详情"
      size="lg"
    >
      <div v-if="currentRole" class="space-y-4">
        <div class="flex items-start gap-4">
          <div class="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
            <svg class="w-6 h-6 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div class="flex-1">
            <h3 class="text-xl font-semibold text-secondary-900">{{ currentRole.name }}</h3>
            <p class="text-sm text-secondary-500">编码：{{ currentRole.code }}</p>
          </div>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="text-sm text-secondary-500">状态</label>
            <Badge :variant="currentRole.status === 'active' ? 'success' : 'danger'">
              {{ currentRole.status === 'active' ? '启用' : '禁用' }}
            </Badge>
          </div>
          <div>
            <label class="text-sm text-secondary-500">权限数量</label>
            <p class="font-medium">{{ currentRole.permissionCount }} 个</p>
          </div>
          <div class="col-span-2">
            <label class="text-sm text-secondary-500">角色描述</label>
            <p class="font-medium">{{ currentRole.description || '-' }}</p>
          </div>
          <div class="col-span-2">
            <label class="text-sm text-secondary-500">创建时间</label>
            <p class="font-medium">{{ formatDate(currentRole.createdAt) }}</p>
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
import Input from '@/components/ui/Input.vue'
import Textarea from '@/components/ui/Textarea.vue'
import Select from '@/components/ui/Select.vue'
import Checkbox from '@/components/ui/Checkbox.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import SearchBar from '@/components/ui/SearchBar.vue'

const loading = ref(false)
const submitting = ref(false)
const showAddModal = ref(false)
const showDetailModal = ref(false)
const showPermissionModal = ref(false)
const editingRole = ref(null)
const currentRole = ref(null)
const searchQuery = ref('')
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const columns = [
  { key: 'id', label: 'ID', width: '80px' },
  { key: 'name', label: '角色名称' },
  { key: 'code', label: '角色编码' },
  { key: 'description', label: '描述', width: '300px' },
  { key: 'permissions', label: '权限' },
  { key: 'status', label: '状态' },
  { key: 'createdAt', label: '创建时间' }
]

const roles = ref([
  { id: 1, name: '超级管理员', code: 'super_admin', description: '拥有系统所有权限', status: 'active', permissionCount: 50, createdAt: '2024-01-01' },
  { id: 2, name: '管理员', code: 'admin', description: '拥有系统管理权限', status: 'active', permissionCount: 30, createdAt: '2024-01-02' },
  { id: 3, name: '普通用户', code: 'user', description: '普通用户权限', status: 'active', permissionCount: 10, createdAt: '2024-01-03' },
  { id: 4, name: '访客', code: 'guest', description: '只读权限', status: 'inactive', permissionCount: 5, createdAt: '2024-01-04' }
])

const roleForm = reactive({
  name: '',
  code: '',
  description: '',
  status: 'active'
})

const formErrors = reactive({
  name: '',
  code: ''
})

const statusOptions = [
  { value: 'active', label: '启用' },
  { value: 'inactive', label: '禁用' }
]

const permissionModules = ref([
  {
    name: '系统管理',
    checked: false,
    permissions: [
      { id: 1, name: '查看用户', checked: false },
      { id: 2, name: '创建用户', checked: false },
      { id: 3, name: '编辑用户', checked: false },
      { id: 4, name: '删除用户', checked: false }
    ]
  },
  {
    name: '金融分析',
    checked: false,
    permissions: [
      { id: 5, name: '查看股票', checked: false },
      { id: 6, name: '股票预测', checked: false },
      { id: 7, name: '风险评估', checked: false }
    ]
  },
  {
    name: '气象分析',
    checked: false,
    permissions: [
      { id: 8, name: '查看气象数据', checked: false },
      { id: 9, name: '气象预测', checked: false }
    ]
  }
])

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const handleSearch = () => {
  page.value = 1
  loadRoles()
}

const handlePageChange = ({ page: newPage, pageSize: newSize }) => {
  page.value = newPage
  pageSize.value = newSize
  loadRoles()
}

const loadRoles = async () => {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
    total.value = roles.value.length
  } finally {
    loading.value = false
  }
}

const viewPermissions = (role) => {
  currentRole.value = role
  showPermissionModal.value = true
}

const editRole = (role) => {
  editingRole.value = role
  Object.assign(roleForm, {
    name: role.name,
    code: role.code,
    description: role.description,
    status: role.status
  })
  showAddModal.value = true
}

const deleteRole = async (role) => {
  if (confirm(`确定要删除角色 ${role.name} 吗？`)) {
    roles.value = roles.value.filter(r => r.id !== role.id)
  }
}

const toggleModulePermissions = (module) => {
  module.permissions.forEach(permission => {
    permission.checked = module.checked
  })
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

const handlePermissionSubmit = async () => {
  submitting.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 1000))
    showPermissionModal.value = false
  } finally {
    submitting.value = false
  }
}

const resetForm = () => {
  editingRole.value = null
  Object.assign(roleForm, {
    name: '',
    code: '',
    description: '',
    status: 'active'
  })
  Object.assign(formErrors, {
    name: '',
    code: ''
  })
}

onMounted(() => {
  loadRoles()
})
</script>
