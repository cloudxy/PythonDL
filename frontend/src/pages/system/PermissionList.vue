<template>
  <div class="space-y-6">
    <PageHeader title="权限管理" subtitle="管理系统权限资源">
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

    <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
      <!-- 权限树 -->
      <Card title="权限结构" class="lg:col-span-1">
        <div class="space-y-2">
          <div
            v-for="group in permissionTree"
            :key="group.id"
            class="border border-secondary-200 rounded-lg overflow-hidden"
          >
            <div
              class="p-3 bg-secondary-50 cursor-pointer flex items-center justify-between"
              @click="toggleGroup(group.id)"
            >
              <span class="font-medium">{{ group.name }}</span>
              <svg
                :class="['w-4 h-4 transition-transform', expandedGroups.includes(group.id) ? 'rotate-180' : '']"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>
            <div v-show="expandedGroups.includes(group.id)" class="p-2 space-y-1">
              <div
                v-for="perm in group.permissions"
                :key="perm.id"
                class="p-2 text-sm text-secondary-600 hover:bg-secondary-50 rounded cursor-pointer"
                :class="{ 'bg-primary-50 text-primary-600': selectedPermission?.id === perm.id }"
                @click="selectPermission(perm)"
              >
                {{ perm.name }}
              </div>
            </div>
          </div>
        </div>
      </Card>

      <!-- 权限列表 -->
      <Card title="权限列表" class="lg:col-span-3">
        <Table
          :columns="columns"
          :data="permissions"
          :loading="loading"
          :show-pagination="false"
        >
          <template #cell-type="{ row }">
            <Badge :variant="row.type === 'menu' ? 'primary' : 'secondary'">
              {{ row.type === 'menu' ? '菜单' : '按钮' }}
            </Badge>
          </template>
          <template #cell-status="{ row }">
            <Badge :variant="row.status === 'active' ? 'success' : 'danger'">
              {{ row.status === 'active' ? '启用' : '禁用' }}
            </Badge>
          </template>
          <template #actions="{ row }">
            <div class="flex items-center justify-end gap-2">
              <Button variant="ghost" size="sm" @click="editPermission(row)">编辑</Button>
              <Button variant="danger" size="sm" @click="deletePermission(row)">删除</Button>
            </div>
          </template>
        </Table>
      </Card>
    </div>

    <!-- 新增/编辑权限模态框 -->
    <Modal
      v-model="showAddModal"
      :title="editingPermission ? '编辑权限' : '新增权限'"
      size="md"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <FormSection>
        <Input
          v-model="permissionForm.name"
          label="权限名称"
          placeholder="请输入权限名称"
          required
        />
        <Input
          v-model="permissionForm.code"
          label="权限标识"
          placeholder="如：user:create"
          required
        />
        <Select
          v-model="permissionForm.type"
          label="权限类型"
          :options="typeOptions"
        />
        <Select
          v-model="permissionForm.parentId"
          label="父级权限"
          :options="parentOptions"
          placeholder="无（顶级权限）"
        />
        <Input
          v-model="permissionForm.path"
          label="路由路径"
          placeholder="如：/system/users"
        />
        <Input
          v-model="permissionForm.icon"
          label="图标"
          placeholder="图标名称"
        />
        <Input
          v-model="permissionForm.sort"
          type="number"
          label="排序"
          placeholder="数字越小越靠前"
        />
        <Select
          v-model="permissionForm.status"
          label="状态"
          :options="statusOptions"
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
import Select from '@/components/ui/Select.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'

const loading = ref(false)
const submitting = ref(false)
const showAddModal = ref(false)
const editingPermission = ref(null)
const selectedPermission = ref(null)
const expandedGroups = ref(['system'])

const columns = [
  { key: 'id', label: 'ID', width: '80px' },
  { key: 'name', label: '权限名称' },
  { key: 'code', label: '权限标识' },
  { key: 'type', label: '类型' },
  { key: 'path', label: '路径' },
  { key: 'sort', label: '排序' },
  { key: 'status', label: '状态' }
]

const permissions = ref([
  { id: 1, name: '用户管理', code: 'user', type: 'menu', path: '/system/users', sort: 1, status: 'active' },
  { id: 2, name: '查看用户', code: 'user:view', type: 'button', path: '', sort: 1, status: 'active' },
  { id: 3, name: '创建用户', code: 'user:create', type: 'button', path: '', sort: 2, status: 'active' },
  { id: 4, name: '编辑用户', code: 'user:edit', type: 'button', path: '', sort: 3, status: 'active' },
  { id: 5, name: '删除用户', code: 'user:delete', type: 'button', path: '', sort: 4, status: 'active' },
  { id: 6, name: '角色管理', code: 'role', type: 'menu', path: '/system/roles', sort: 2, status: 'active' }
])

const permissionTree = ref([
  {
    id: 'system',
    name: '系统管理',
    permissions: [
      { id: 1, name: '用户管理' },
      { id: 6, name: '角色管理' }
    ]
  },
  {
    id: 'finance',
    name: '金融分析',
    permissions: [
      { id: 7, name: '股票管理' },
      { id: 8, name: '风险评估' }
    ]
  }
])

const permissionForm = reactive({
  name: '',
  code: '',
  type: 'menu',
  parentId: '',
  path: '',
  icon: '',
  sort: 0,
  status: 'active'
})

const typeOptions = [
  { value: 'menu', label: '菜单' },
  { value: 'button', label: '按钮' }
]

const parentOptions = [
  { value: 'system', label: '系统管理' },
  { value: 'finance', label: '金融分析' },
  { value: 'weather', label: '气象分析' }
]

const statusOptions = [
  { value: 'active', label: '启用' },
  { value: 'inactive', label: '禁用' }
]

const toggleGroup = (groupId) => {
  const index = expandedGroups.value.indexOf(groupId)
  if (index > -1) {
    expandedGroups.value.splice(index, 1)
  } else {
    expandedGroups.value.push(groupId)
  }
}

const selectPermission = (perm) => {
  selectedPermission.value = perm
}

const editPermission = (perm) => {
  editingPermission.value = perm
  Object.assign(permissionForm, perm)
  showAddModal.value = true
}

const deletePermission = async (perm) => {
  if (confirm(`确定要删除权限 ${perm.name} 吗？`)) {
    permissions.value = permissions.value.filter(p => p.id !== perm.id)
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
    type: 'menu',
    parentId: '',
    path: '',
    icon: '',
    sort: 0,
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
