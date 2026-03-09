<template>
  <div class="space-y-6">
    <PageHeader title="用户管理" subtitle="管理系统用户账户">
      <template #actions>
        <Button variant="primary" @click="showAddModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          新增用户
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '系统管理' },
      { label: '用户管理' }
    ]" />

    <Card>
      <SearchBar
        v-model="searchQuery"
        placeholder="搜索用户名、邮箱..."
        @search="handleSearch"
        @add="showAddModal = true"
      />

      <Table
        :columns="columns"
        :data="users"
        :loading="loading"
        :total="total"
        :page="page"
        :page-size="pageSize"
        @page-change="handlePageChange"
      >
        <template #cell-status="{ row }">
          <Badge :variant="row.status === 'active' ? 'success' : 'danger'">
            {{ row.status === 'active' ? '正常' : '禁用' }}
          </Badge>
        </template>
        <template #cell-role="{ row }">
          <Badge variant="primary">{{ row.role }}</Badge>
        </template>
        <template #cell-lastLogin="{ row }">
          {{ formatDate(row.lastLogin) }}
        </template>
        <template #cell-createdAt="{ row }">
          {{ formatDate(row.createdAt) }}
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button variant="ghost" size="sm" @click="viewUser(row)">查看</Button>
            <Button variant="ghost" size="sm" @click="editUser(row)">编辑</Button>
            <Button variant="ghost" size="sm" @click="resetPassword(row)">重置密码</Button>
            <Button variant="danger" size="sm" @click="deleteUser(row)">删除</Button>
          </div>
        </template>
      </Table>
    </Card>

    <!-- 新增/编辑用户模态框 -->
    <Modal
      v-model="showAddModal"
      :title="editingUser ? '编辑用户' : '新增用户'"
      size="lg"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <FormSection title="基本信息" description="请填写用户的基本信息">
        <Input
          v-model="userForm.username"
          label="用户名"
          placeholder="请输入用户名"
          :error="formErrors.username"
          required
        />
        <Input
          v-model="userForm.email"
          type="email"
          label="邮箱"
          placeholder="请输入邮箱"
          :error="formErrors.email"
          required
        />
        <PasswordInput
          v-if="!editingUser"
          v-model="userForm.password"
          label="密码"
          placeholder="请输入密码"
          :error="formErrors.password"
          required
        />
        <Input
          v-model="userForm.phone"
          label="手机号"
          placeholder="请输入手机号"
        />
      </FormSection>

      <FormSection title="角色权限" description="设置用户的角色和权限">
        <Select
          v-model="userForm.roleId"
          label="角色"
          :options="roleOptions"
          placeholder="请选择角色"
          required
        />
        <Select
          v-model="userForm.status"
          label="状态"
          :options="statusOptions"
          placeholder="请选择状态"
        />
      </FormSection>
    </Modal>

    <!-- 用户详情模态框 -->
    <Modal
      v-model="showDetailModal"
      title="用户详情"
      size="lg"
    >
      <div v-if="currentUser" class="space-y-4">
        <div class="flex items-center gap-4">
          <Avatar :name="currentUser.username" size="xl" />
          <div>
            <h3 class="text-lg font-semibold">{{ currentUser.username }}</h3>
            <p class="text-secondary-500">{{ currentUser.email }}</p>
          </div>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="text-sm text-secondary-500">手机号</label>
            <p class="font-medium">{{ currentUser.phone || '-' }}</p>
          </div>
          <div>
            <label class="text-sm text-secondary-500">角色</label>
            <p class="font-medium">{{ currentUser.role }}</p>
          </div>
          <div>
            <label class="text-sm text-secondary-500">状态</label>
            <Badge :variant="currentUser.status === 'active' ? 'success' : 'danger'">
              {{ currentUser.status === 'active' ? '正常' : '禁用' }}
            </Badge>
          </div>
          <div>
            <label class="text-sm text-secondary-500">最后登录</label>
            <p class="font-medium">{{ formatDate(currentUser.lastLogin) }}</p>
          </div>
          <div>
            <label class="text-sm text-secondary-500">创建时间</label>
            <p class="font-medium">{{ formatDate(currentUser.createdAt) }}</p>
          </div>
          <div>
            <label class="text-sm text-secondary-500">更新时间</label>
            <p class="font-medium">{{ formatDate(currentUser.updatedAt) }}</p>
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
import PasswordInput from '@/components/ui/PasswordInput.vue'
import Select from '@/components/ui/Select.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import SearchBar from '@/components/ui/SearchBar.vue'
import Avatar from '@/components/ui/Avatar.vue'

const loading = ref(false)
const submitting = ref(false)
const showAddModal = ref(false)
const showDetailModal = ref(false)
const editingUser = ref(null)
const currentUser = ref(null)
const searchQuery = ref('')
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const columns = [
  { key: 'id', label: 'ID', width: '80px' },
  { key: 'username', label: '用户名' },
  { key: 'email', label: '邮箱' },
  { key: 'phone', label: '手机号' },
  { key: 'role', label: '角色' },
  { key: 'status', label: '状态' },
  { key: 'lastLogin', label: '最后登录' },
  { key: 'createdAt', label: '创建时间' }
]

const users = ref([
  { id: 1, username: 'admin', email: 'admin@example.com', phone: '13800138000', role: '管理员', status: 'active', lastLogin: '2024-01-15 10:30:00', createdAt: '2024-01-01' },
  { id: 2, username: 'user1', email: 'user1@example.com', phone: '13800138001', role: '普通用户', status: 'active', lastLogin: '2024-01-14 15:20:00', createdAt: '2024-01-02' },
  { id: 3, username: 'user2', email: 'user2@example.com', phone: '13800138002', role: '普通用户', status: 'inactive', lastLogin: '2024-01-10 09:15:00', createdAt: '2024-01-03' }
])

const userForm = reactive({
  username: '',
  email: '',
  password: '',
  phone: '',
  roleId: '',
  status: 'active'
})

const formErrors = reactive({
  username: '',
  email: '',
  password: ''
})

const roleOptions = [
  { value: '1', label: '管理员' },
  { value: '2', label: '普通用户' },
  { value: '3', label: '访客' }
]

const statusOptions = [
  { value: 'active', label: '正常' },
  { value: 'inactive', label: '禁用' }
]

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const handleSearch = () => {
  page.value = 1
  loadUsers()
}

const handlePageChange = ({ page: newPage, pageSize: newSize }) => {
  page.value = newPage
  pageSize.value = newSize
  loadUsers()
}

const loadUsers = async () => {
  loading.value = true
  try {
    // TODO: 调用 API
    await new Promise(resolve => setTimeout(resolve, 500))
    total.value = users.value.length
  } finally {
    loading.value = false
  }
}

const viewUser = (user) => {
  currentUser.value = user
  showDetailModal.value = true
}

const editUser = (user) => {
  editingUser.value = user
  Object.assign(userForm, {
    username: user.username,
    email: user.email,
    phone: user.phone,
    roleId: '1',
    status: user.status
  })
  showAddModal.value = true
}

const resetPassword = async (user) => {
  if (confirm(`确定要重置用户 ${user.username} 的密码吗？`)) {
    // TODO: 调用 API
    alert('密码已重置')
  }
}

const deleteUser = async (user) => {
  if (confirm(`确定要删除用户 ${user.username} 吗？`)) {
    // TODO: 调用 API
    users.value = users.value.filter(u => u.id !== user.id)
  }
}

const handleSubmit = async () => {
  submitting.value = true
  try {
    // TODO: 调用 API
    await new Promise(resolve => setTimeout(resolve, 1000))
    showAddModal.value = false
    resetForm()
  } finally {
    submitting.value = false
  }
}

const resetForm = () => {
  editingUser.value = null
  Object.assign(userForm, {
    username: '',
    email: '',
    password: '',
    phone: '',
    roleId: '',
    status: 'active'
  })
  Object.assign(formErrors, {
    username: '',
    email: '',
    password: ''
  })
}

onMounted(() => {
  loadUsers()
})
</script>
