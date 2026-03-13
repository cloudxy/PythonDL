<template>
  <div class="space-y-6">
    <PageHeader title="气象站点" subtitle="管理气象监测站点">
      <template #actions>
        <Button variant="primary" @click="showAddModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          新增站点
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '气象分析' },
      { label: '气象站点' }
    ]" />

    <Card>
      <SearchBar
        v-model="searchQuery"
        placeholder="搜索站点名称、编码..."
        @search="handleSearch"
        @add="showAddModal = true"
      />

      <Table
        :columns="columns"
        :data="stations"
        :loading="loading"
        :total="total"
        :page="page"
        :page-size="pageSize"
        @page-change="handlePageChange"
      >
        <template #cell-status="{ row }">
          <Badge :variant="row.status === 'active' ? 'success' : 'secondary'">
            {{ row.status === 'active' ? '运行中' : '离线' }}
          </Badge>
        </template>
        <template #cell-type="{ row }">
          <Badge :variant="getTypeBadgeVariant(row.type)">
            {{ typeMap[row.type] }}
          </Badge>
        </template>
        <template #cell-lastUpdate="{ row }">
          {{ formatDate(row.lastUpdate) }}
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button variant="ghost" size="sm" @click="viewStation(row)">详情</Button>
            <Button variant="ghost" size="sm" @click="editStation(row)">编辑</Button>
            <Button variant="danger" size="sm" @click="deleteStation(row)">删除</Button>
          </div>
        </template>
      </Table>
    </Card>

    <!-- 新增/编辑站点模态框 -->
    <Modal
      v-model="showAddModal"
      :title="editingStation ? '编辑站点' : '新增站点'"
      size="lg"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <FormSection title="基本信息">
        <Input
          v-model="stationForm.name"
          label="站点名称"
          placeholder="请输入站点名称"
          :error="formErrors.name"
          required
        />
        <Input
          v-model="stationForm.code"
          label="站点编码"
          placeholder="请输入站点编码"
          :error="formErrors.code"
          :disabled="!!editingStation"
          required
        />
        <Input
          v-model="stationForm.location"
          label="地理位置"
          placeholder="请输入站点位置"
          required
        />
        <div class="grid grid-cols-2 gap-4">
          <Input
            v-model="stationForm.latitude"
            label="纬度"
            placeholder="如：39.9042"
            type="number"
            step="0.0001"
          />
          <Input
            v-model="stationForm.longitude"
            label="经度"
            placeholder="如：116.4074"
            type="number"
            step="0.0001"
          />
        </div>
        <Select
          v-model="stationForm.type"
          label="站点类型"
          :options="typeOptions"
          placeholder="请选择站点类型"
        />
        <Select
          v-model="stationForm.status"
          label="状态"
          :options="statusOptions"
          placeholder="请选择状态"
        />
      </FormSection>
    </Modal>

    <!-- 站点详情模态框 -->
    <Modal
      v-model="showDetailModal"
      title="站点详情"
      size="lg"
    >
      <div v-if="currentStation" class="space-y-4">
        <div class="flex items-start gap-4">
          <div class="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
            <svg class="w-6 h-6 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <div class="flex-1">
            <h3 class="text-xl font-semibold text-secondary-900">{{ currentStation.name }}</h3>
            <p class="text-sm text-secondary-500">编码：{{ currentStation.code }}</p>
          </div>
          <Badge :variant="currentStation.status === 'active' ? 'success' : 'secondary'">
            {{ currentStation.status === 'active' ? '运行中' : '离线' }}
          </Badge>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="text-sm text-secondary-500">地理位置</label>
            <p class="font-medium">{{ currentStation.location }}</p>
          </div>
          <div>
            <label class="text-sm text-secondary-500">站点类型</label>
            <Badge :variant="getTypeBadgeVariant(currentStation.type)">
              {{ typeMap[currentStation.type] }}
            </Badge>
          </div>
          <div>
            <label class="text-sm text-secondary-500">坐标</label>
            <p class="font-medium">{{ currentStation.latitude }}, {{ currentStation.longitude }}</p>
          </div>
          <div>
            <label class="text-sm text-secondary-500">海拔</label>
            <p class="font-medium">{{ currentStation.altitude || '-' }} 米</p>
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
import Select from '@/components/ui/Select.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import SearchBar from '@/components/ui/SearchBar.vue'

const loading = ref(false)
const submitting = ref(false)
const showAddModal = ref(false)
const showDetailModal = ref(false)
const editingStation = ref(null)
const searchQuery = ref('')
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const typeMap = {
  national: '国家站',
  regional: '区域站',
  automatic: '自动站',
  radar: '雷达站'
}

const columns = [
  { key: 'code', label: '站点编码' },
  { key: 'name', label: '站点名称' },
  { key: 'location', label: '地理位置', width: '200px' },
  { key: 'type', label: '类型' },
  { key: 'status', label: '状态' },
  { key: 'lastUpdate', label: '更新时间' }
]

const stations = ref([
  { id: 1, name: '北京国家气象站', code: 'BJ001', location: '北京市海淀区', type: 'national', status: 'active', latitude: 39.9042, longitude: 116.4074, altitude: 50, lastUpdate: '2024-01-15 10:30:00' },
  { id: 2, name: '上海浦东气象站', code: 'SH001', location: '上海市浦东新区', type: 'regional', status: 'active', latitude: 31.2304, longitude: 121.4737, altitude: 4, lastUpdate: '2024-01-15 10:25:00' },
  { id: 3, name: '广州天河气象站', code: 'GZ001', location: '广州市天河区', type: 'automatic', status: 'active', latitude: 23.1291, longitude: 113.2644, altitude: 10, lastUpdate: '2024-01-15 10:20:00' },
  { id: 4, name: '成都双流气象站', code: 'CD001', location: '成都市双流区', type: 'regional', status: 'inactive', latitude: 30.5728, longitude: 104.0668, altitude: 500, lastUpdate: '2024-01-14 18:00:00' }
])

const stationForm = reactive({
  name: '',
  code: '',
  location: '',
  latitude: '',
  longitude: '',
  type: 'automatic',
  status: 'active'
})

const formErrors = reactive({
  name: '',
  code: ''
})

const typeOptions = [
  { value: 'national', label: '国家站' },
  { value: 'regional', label: '区域站' },
  { value: 'automatic', label: '自动站' },
  { value: 'radar', label: '雷达站' }
]

const statusOptions = [
  { value: 'active', label: '运行中' },
  { value: 'inactive', label: '离线' }
]

const currentStation = ref(null)

const getTypeBadgeVariant = (type) => {
  const variants = {
    national: 'primary',
    regional: 'info',
    automatic: 'success',
    radar: 'warning'
  }
  return variants[type] || 'secondary'
}

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const handleSearch = () => {
  page.value = 1
  loadStations()
}

const handlePageChange = ({ page: newPage, pageSize: newSize }) => {
  page.value = newPage
  pageSize.value = newSize
  loadStations()
}

const loadStations = async () => {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
    total.value = stations.value.length
  } finally {
    loading.value = false
  }
}

const viewStation = (station) => {
  currentStation.value = station
  showDetailModal.value = true
}

const editStation = (station) => {
  editingStation.value = station
  Object.assign(stationForm, {
    name: station.name,
    code: station.code,
    location: station.location,
    latitude: station.latitude.toString(),
    longitude: station.longitude.toString(),
    type: station.type,
    status: station.status
  })
  showAddModal.value = true
}

const deleteStation = async (station) => {
  if (confirm(`确定要删除站点 ${station.name} 吗？`)) {
    stations.value = stations.value.filter(s => s.id !== station.id)
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
  editingStation.value = null
  Object.assign(stationForm, {
    name: '',
    code: '',
    location: '',
    latitude: '',
    longitude: '',
    type: 'automatic',
    status: 'active'
  })
  Object.assign(formErrors, {
    name: '',
    code: ''
  })
}

onMounted(() => {
  loadStations()
})
</script>
