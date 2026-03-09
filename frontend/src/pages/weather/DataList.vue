<template>
  <div class="space-y-6">
    <PageHeader title="气象数据" subtitle="查看和分析气象监测数据" />

    <Breadcrumb :items="[
      { label: '气象分析' },
      { label: '气象数据' }
    ]" />

    <!-- 筛选条件 -->
    <Card>
      <div class="flex flex-wrap items-center gap-4">
        <Select
          v-model="filters.stationId"
          :options="stationOptions"
          placeholder="选择站点"
        />
        <DatePicker v-model="filters.startDate" placeholder="开始日期" />
        <DatePicker v-model="filters.endDate" placeholder="结束日期" />
        <Button variant="primary" @click="loadData">查询</Button>
        <Button variant="secondary" @click="exportData">导出数据</Button>
      </div>
    </Card>

    <!-- 数据概览 -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
      <StatCard label="平均温度" value="23.5" suffix="°C" color="warning" />
      <StatCard label="平均湿度" value="65" suffix="%" color="primary" />
      <StatCard label="平均风速" value="3.2" suffix="m/s" color="success" />
      <StatCard label="降水量" value="12.5" suffix="mm" color="danger" />
    </div>

    <!-- 数据表格 -->
    <Card title="气象数据列表">
      <Table
        :columns="columns"
        :data="weatherData"
        :loading="loading"
        :total="total"
        :page="page"
        :page-size="pageSize"
        @page-change="handlePageChange"
      >
        <template #cell-temperature="{ row }">
          <span :class="getTempClass(row.temperature)">{{ row.temperature }}°C</span>
        </template>
        <template #cell-weather="{ row }">
          <Badge :variant="getWeatherVariant(row.weather)">{{ row.weather }}</Badge>
        </template>
        <template #cell-recordTime="{ row }">
          {{ formatDate(row.recordTime) }}
        </template>
      </Table>
    </Card>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Select from '@/components/ui/Select.vue'
import DatePicker from '@/components/ui/DatePicker.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import StatCard from '@/components/ui/StatCard.vue'

const loading = ref(false)
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const filters = reactive({
  stationId: '',
  startDate: '',
  endDate: ''
})

const columns = [
  { key: 'stationName', label: '站点' },
  { key: 'recordTime', label: '记录时间' },
  { key: 'temperature', label: '温度' },
  { key: 'humidity', label: '湿度(%)' },
  { key: 'windSpeed', label: '风速(m/s)' },
  { key: 'windDirection', label: '风向' },
  { key: 'precipitation', label: '降水量(mm)' },
  { key: 'weather', label: '天气' }
]

const weatherData = ref([
  { id: 1, stationName: '北京气象站', recordTime: '2024-01-15 10:00:00', temperature: 5, humidity: 45, windSpeed: 3.2, windDirection: '东北风', precipitation: 0, weather: '晴' },
  { id: 2, stationName: '北京气象站', recordTime: '2024-01-15 11:00:00', temperature: 7, humidity: 42, windSpeed: 3.5, windDirection: '东北风', precipitation: 0, weather: '晴' },
  { id: 3, stationName: '上海气象站', recordTime: '2024-01-15 10:00:00', temperature: 12, humidity: 65, windSpeed: 2.8, windDirection: '东风', precipitation: 0.5, weather: '多云' }
])

const stationOptions = [
  { value: '1', label: '北京气象站' },
  { value: '2', label: '上海气象站' },
  { value: '3', label: '广州气象站' }
]

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const getTempClass = (temp) => {
  if (temp < 0) return 'text-blue-600'
  if (temp < 15) return 'text-cyan-600'
  if (temp < 25) return 'text-green-600'
  if (temp < 35) return 'text-orange-600'
  return 'text-red-600'
}

const getWeatherVariant = (weather) => {
  const variants = {
    '晴': 'success',
    '多云': 'primary',
    '阴': 'secondary',
    '小雨': 'primary',
    '大雨': 'warning',
    '暴雨': 'danger'
  }
  return variants[weather] || 'secondary'
}

const handlePageChange = ({ page: newPage }) => {
  page.value = newPage
}

const loadData = () => {
  loading.value = true
  setTimeout(() => {
    loading.value = false
    total.value = weatherData.value.length
  }, 500)
}

const exportData = () => {
  alert('数据导出中...')
}

onMounted(() => {
  loadData()
})
</script>
