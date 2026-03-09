<template>
  <div class="space-y-6">
    <PageHeader title="角色管理" subtitle="管理系统角色和权限">
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
      <Table
        :columns="columns"
        :data="roles"
        :loading="loading"
        :show-pagination="false"
      >
        <template #cell-permissions="{ row }">
          <div class="flex flex-wrap gap-1">
            <Badge v-for="perm in row.permissions.slice(0, 3)" :key="perm" variant="secondary" size="sm">
              {{ perm }}
            </Badge>
            <Badge v-if="row.permissions.length > 3" variant="secondary" size="sm">
              +{{ row.permissions.length - 3 }}
            </Badge>
          </div>
        </template>
        <template #cell-status="{ row }">
          <Badge :variant="row.status === 'active' ? 'success' : 'danger'">
            {{ row.status === 'active' ? '启用' : '禁用' }}
          </Badge>
        </template>
        <template #cell-createdAt="{ row }">
          {{ formatDate(row.createdAt) }}
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button variant="ghost" size="sm" @click="editRole(row)">编辑</Button>
            <Button variant="ghost" size="sm" @click="configPermissions(row)">配置权限</Button>
            <Button variant="danger" size="sm" @click="deleteRole(row)">删除</Button>
          </div>
        </template>
      </Table>
    </Card>

    <!-- 新增/编辑角色模态框 -->
    <Modal
      v-model="showAddModal"
      :title="editingRole ? '编辑角色' : '新增角色'"
      size="md"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <FormSection>
        <Input
          v-model="roleForm.name"
          label="角色名称"
          placeholder="请输入角色名称"
          :error="formErrors.name"
          required
        />
        <Input
          v-model="roleForm.code"
          label="角色标识"
          placeholder="请输入角色标识"
          :error="formErrors.code"
          hint="如：admin, user, guest"
          required
        />
        <Textarea
          v-model="roleForm.description"
          label="角色描述"
          placeholder="请输入角色描述"
        />
        <Select
          v-model="roleForm.status"
          label="状态"
          :options="statusOptions"
        />
      </FormSection>
    </Modal>

    <!-- 权限配置模态框 -->
    <Modal
      v-model="showPermissionModal"
      title="配置权限"
      size="lg"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handlePermissionSubmit"
    >
      <div class="space-y-4">
        <div v-for="group in permissionGroups" :key="group.name" class="border border-secondary-200 rounded-lg p-4">
          <h4 class="font-medium text-secondary-900 mb-3">{{ group.label }}</h4>
          <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
            <Checkbox
              v-for="permission in group.permissions"
              :key="permission.id"
              v-model="selectedPermissions"
              :value="permission.id"
              :label="permission.name"
            />
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
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import Checkbox from '@/components/ui/Checkbox.vue'

const loading = ref(false)
const submitting = ref(false)
const showAddModal = ref(false)
const showPermissionModal = ref(false)
const editingRole = ref(null)
const currentRole = ref(null)
const selectedPermissions = ref([])

const columns = [
  { key: 'id', label: 'ID', width: '80px' },
  { key: 'name', label: '角色名称' },
  { key: 'code', label: '角色标识' },
  { key: 'description', label: '描述' },
  { key: 'permissions', label: '权限' },
  { key: 'userCount', label: '用户数' },
  { key: 'status', label: '状态' },
  { key: 'createdAt', label: '创建时间' }
]

const roles = ref([
  { id: 1, name: '超级管理员', code: 'super_admin', description: '拥有系统所有权限', permissions: ['用户管理', '角色管理', '权限管理', '系统配置'], userCount: 1, status: 'active', createdAt: '2024-01-01' },
  { id: 2, name: '管理员', code: 'admin', description: '拥有大部分管理权限', permissions: ['用户管理', '数据查看', '报表导出'], userCount: 5, status: 'active', createdAt: '2024-01-02' },
  { id: 3, name: '普通用户', code: 'user', description: '普通用户权限', permissions: ['数据查看'], userCount: 100, status: 'active', createdAt: '2024-01-03' }
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

const permissionGroups = ref([
  {
    name: 'system',
    label: '系统管理',
    permissions: [
      { id: 'user:view', name: '查看用户' },
      { id: 'user:create', name: '创建用户' },
      { id: 'user:edit', name: '编辑用户' },
      { id: 'user:delete', name: '删除用户' },
      { id: 'role:view', name: '查看角色' },
      { id: 'role:create', name: '创建角色' },
      { id: 'role:edit', name: '编辑角色' },
      { id: 'role:delete', name: '删除角色' }
    ]
  },
  {
    name: 'finance',
    label: '金融分析',
    permissions: [
      { id: 'stock:view', name: '查看股票' },
      { id: 'stock:create', name: '添加股票' },
      { id: 'stock:predict', name: '股票预测' },
      { id: 'risk:assess', name: '风险评估' }
    ]
  },
  {
    name: 'weather',
    label: '气象分析',
    permissions: [
      { id: 'weather:view', name: '查看气象数据' },
      { id: 'weather:predict', name: '气象预测' }
    ]
  }
])

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleDateString('zh-CN')
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

const configPermissions = (role) => {
  currentRole.value = role
  selectedPermissions.value = ['user:view', 'stock:view']
  showPermissionModal.value = true
}

const deleteRole = async (role) => {
  if (confirm(`确定要删除角色 ${role.name} 吗？`)) {
    roles.value = roles.value.filter(r => r.id !== role.id)
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
}

onMounted(() => {
  loading.value = true
  setTimeout(() => {
    loading.value = false
  }, 500)
})
</script>
