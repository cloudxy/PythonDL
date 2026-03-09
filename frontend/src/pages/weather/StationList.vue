<template>
  <div class="space-y-6">
    <PageHeader title="气象站点管理" subtitle="管理气象监测站点信息">
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
          <StatusIndicator :status="row.status === 'active' ? 'success' : 'danger'" :text="row.status === 'active' ? '在线' : '离线'" />
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
        <Input v-model="stationForm.name" label="站点名称" placeholder="请输入站点名称" required />
        <Input v-model="stationForm.code" label="站点代码" placeholder="请输入站点代码" required />
        <Select v-model="stationForm.type" label="站点类型" :options="typeOptions" />
      </FormSection>
      <FormSection title="位置信息">
        <Input v-model="stationForm.province" label="省份" placeholder="请输入省份" />
        <Input v-model="stationForm.city" label="城市" placeholder="请输入城市" />
        <Input v-model="stationForm.latitude" type="number" label="纬度" placeholder="如：39.9042" />
        <Input v-model="stationForm.longitude" type="number" label="经度" placeholder="如：116.4074" />
        <Input v-model="stationForm.altitude" type="number" label="海拔(米)" placeholder="请输入海拔" />
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
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import StatusIndicator from '@/components/ui/StatusIndicator.vue'

const loading = ref(false)
const submitting = ref(false)
const showAddModal = ref(false)
const editingStation = ref(null)
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const columns = [
  { key: 'id', label: 'ID', width: '80px' },
  { key: 'name', label: '站点名称' },
  { key: 'code', label: '站点代码' },
  { key: 'city', label: '城市' },
  { key: 'type', label: '类型' },
  { key: 'status', label: '状态' },
  { key: 'lastUpdate', label: '最后更新' }
]

const stations = ref([
  { id: 1, name: '北京气象站', code: 'BJ001', city: '北京', type: '国家站', status: 'active', lastUpdate: '2024-01-15 10:00:00' },
  { id: 2, name: '上海气象站', code: 'SH001', city: '上海', type: '国家站', status: 'active', lastUpdate: '2024-01-15 10:00:00' },
  { id: 3, name: '广州气象站', code: 'GZ001', city: '广州', type: '区域站', status: 'active', lastUpdate: '2024-01-15 10:00:00' }
])

const stationForm = reactive({
  name: '',
  code: '',
  type: 'national',
  province: '',
  city: '',
  latitude: '',
  longitude: '',
  altitude: ''
})

const typeOptions = [
  { value: 'national', label: '国家站' },
  { value: 'regional', label: '区域站' },
  { value: 'auto', label: '自动站' }
]

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const handlePageChange = ({ page: newPage }) => {
  page.value = newPage
}

const viewStation = (station) => {
  // TODO: 跳转到详情页
}

const editStation = (station) => {
  editingStation.value = station
  Object.assign(stationForm, station)
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
  } finally {
    submitting.value = false
  }
}

onMounted(() => {
  loading.value = true
  setTimeout(() => {
    loading.value = false
    total.value = stations.value.length
  }, 500)
})
</script>
