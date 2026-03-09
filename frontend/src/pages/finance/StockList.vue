<template>
  <div class="space-y-6">
    <PageHeader title="股票管理" subtitle="管理股票基础信息">
      <template #actions>
        <Button variant="primary" @click="showAddModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          新增股票
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '金融分析' },
      { label: '股票管理' }
    ]" />

    <Card>
      <SearchBar
        v-model="searchQuery"
        placeholder="搜索股票代码、名称..."
        @search="handleSearch"
        @add="showAddModal = true"
      />

      <Table
        :columns="columns"
        :data="stocks"
        :loading="loading"
        :total="total"
        :page="page"
        :page-size="pageSize"
        @page-change="handlePageChange"
      >
        <template #cell-price="{ row }">
          <span :class="row.change >= 0 ? 'text-success-600' : 'text-danger-600'">
            {{ row.price.toFixed(2) }}
          </span>
        </template>
        <template #cell-change="{ row }">
          <span :class="row.change >= 0 ? 'text-success-600' : 'text-danger-600'">
            {{ row.change >= 0 ? '+' : '' }}{{ row.change.toFixed(2) }}%
          </span>
        </template>
        <template #cell-status="{ row }">
          <Badge :variant="row.status === 'trading' ? 'success' : 'secondary'">
            {{ row.status === 'trading' ? '交易中' : '停牌' }}
          </Badge>
        </template>
        <template #cell-market="{ row }">
          <Badge variant="primary">{{ row.market }}</Badge>
        </template>
        <template #cell-lastUpdate="{ row }">
          {{ formatDate(row.lastUpdate) }}
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button variant="ghost" size="sm" @click="viewStock(row)">详情</Button>
            <Button variant="ghost" size="sm" @click="editStock(row)">编辑</Button>
            <Button variant="ghost" size="sm" @click="syncData(row)">同步</Button>
            <Button variant="danger" size="sm" @click="deleteStock(row)">删除</Button>
          </div>
        </template>
      </Table>
    </Card>

    <!-- 新增/编辑股票模态框 -->
    <Modal
      v-model="showAddModal"
      :title="editingStock ? '编辑股票' : '新增股票'"
      size="lg"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <FormSection title="基本信息">
        <Input v-model="stockForm.code" label="股票代码" placeholder="如：000001" :disabled="!!editingStock" required />
        <Input v-model="stockForm.name" label="股票名称" placeholder="如：平安银行" required />
        <Select v-model="stockForm.market" label="所属市场" :options="marketOptions" required />
        <Select v-model="stockForm.industry" label="所属行业" :options="industryOptions" />
      </FormSection>
      <FormSection title="其他信息">
        <Input v-model="stockForm.listDate" type="date" label="上市日期" />
        <Textarea v-model="stockForm.description" label="公司简介" placeholder="请输入公司简介" />
        <Select v-model="stockForm.status" label="状态" :options="statusOptions" />
      </FormSection>
    </Modal>

    <!-- 股票详情模态框 -->
    <Modal v-model="showDetailModal" title="股票详情" size="xl">
      <div v-if="currentStock" class="space-y-6">
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-2xl font-bold">{{ currentStock.name }}</h3>
            <p class="text-secondary-500">{{ currentStock.code }} · {{ currentStock.market }}</p>
          </div>
          <div class="text-right">
            <p class="text-3xl font-bold" :class="currentStock.change >= 0 ? 'text-success-600' : 'text-danger-600'">
              {{ currentStock.price.toFixed(2) }}
            </p>
            <p :class="currentStock.change >= 0 ? 'text-success-600' : 'text-danger-600'">
              {{ currentStock.change >= 0 ? '+' : '' }}{{ currentStock.change.toFixed(2) }}%
            </p>
          </div>
        </div>

        <div class="grid grid-cols-4 gap-4">
          <div class="p-4 bg-secondary-50 rounded-lg">
            <p class="text-sm text-secondary-500">今开</p>
            <p class="text-lg font-semibold">{{ currentStock.open?.toFixed(2) || '-' }}</p>
          </div>
          <div class="p-4 bg-secondary-50 rounded-lg">
            <p class="text-sm text-secondary-500">最高</p>
            <p class="text-lg font-semibold text-success-600">{{ currentStock.high?.toFixed(2) || '-' }}</p>
          </div>
          <div class="p-4 bg-secondary-50 rounded-lg">
            <p class="text-sm text-secondary-500">最低</p>
            <p class="text-lg font-semibold text-danger-600">{{ currentStock.low?.toFixed(2) || '-' }}</p>
          </div>
          <div class="p-4 bg-secondary-50 rounded-lg">
            <p class="text-sm text-secondary-500">成交量</p>
            <p class="text-lg font-semibold">{{ formatVolume(currentStock.volume) }}</p>
          </div>
        </div>

        <div class="h-64 bg-secondary-50 rounded-lg flex items-center justify-center text-secondary-400">
          <p>K线图区域（集成图表库后显示）</p>
        </div>

        <div class="grid grid-cols-2 gap-6">
          <div>
            <h4 class="font-semibold mb-3">公司信息</h4>
            <div class="space-y-2 text-sm">
              <p><span class="text-secondary-500">行业：</span>{{ currentStock.industry }}</p>
              <p><span class="text-secondary-500">上市日期：</span>{{ currentStock.listDate }}</p>
              <p><span class="text-secondary-500">状态：</span>
                <Badge :variant="currentStock.status === 'trading' ? 'success' : 'secondary'">
                  {{ currentStock.status === 'trading' ? '交易中' : '停牌' }}
                </Badge>
              </p>
            </div>
          </div>
          <div>
            <h4 class="font-semibold mb-3">公司简介</h4>
            <p class="text-sm text-secondary-600">{{ currentStock.description || '暂无简介' }}</p>
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
import SearchBar from '@/components/ui/SearchBar.vue'

const loading = ref(false)
const submitting = ref(false)
const showAddModal = ref(false)
const showDetailModal = ref(false)
const editingStock = ref(null)
const currentStock = ref(null)
const searchQuery = ref('')
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const columns = [
  { key: 'code', label: '股票代码' },
  { key: 'name', label: '股票名称' },
  { key: 'market', label: '市场' },
  { key: 'price', label: '最新价' },
  { key: 'change', label: '涨跌幅' },
  { key: 'industry', label: '行业' },
  { key: 'status', label: '状态' },
  { key: 'lastUpdate', label: '更新时间' }
]

const stocks = ref([
  { id: 1, code: '000001', name: '平安银行', market: '深交所', price: 12.35, change: 2.15, open: 12.10, high: 12.50, low: 12.05, volume: 125000000, industry: '银行', status: 'trading', lastUpdate: '2024-01-15 15:00:00' },
  { id: 2, code: '000002', name: '万科A', market: '深交所', price: 8.56, change: -1.23, open: 8.70, high: 8.75, low: 8.50, volume: 89000000, industry: '房地产', status: 'trading', lastUpdate: '2024-01-15 15:00:00' },
  { id: 3, code: '600000', name: '浦发银行', market: '上交所', price: 7.89, change: 0.85, open: 7.85, high: 7.95, low: 7.80, volume: 67000000, industry: '银行', status: 'trading', lastUpdate: '2024-01-15 15:00:00' }
])

const stockForm = reactive({
  code: '',
  name: '',
  market: '',
  industry: '',
  listDate: '',
  description: '',
  status: 'trading'
})

const marketOptions = [
  { value: 'sh', label: '上交所' },
  { value: 'sz', label: '深交所' },
  { value: 'bj', label: '北交所' }
]

const industryOptions = [
  { value: 'bank', label: '银行' },
  { value: 'realestate', label: '房地产' },
  { value: 'tech', label: '科技' },
  { value: 'finance', label: '金融' }
]

const statusOptions = [
  { value: 'trading', label: '交易中' },
  { value: 'suspended', label: '停牌' }
]

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const formatVolume = (volume) => {
  if (!volume) return '-'
  if (volume >= 100000000) return (volume / 100000000).toFixed(2) + '亿'
  if (volume >= 10000) return (volume / 10000).toFixed(2) + '万'
  return volume.toString()
}

const handleSearch = () => {
  page.value = 1
  loadStocks()
}

const handlePageChange = ({ page: newPage }) => {
  page.value = newPage
  loadStocks()
}

const loadStocks = async () => {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
    total.value = stocks.value.length
  } finally {
    loading.value = false
  }
}

const viewStock = (stock) => {
  currentStock.value = stock
  showDetailModal.value = true
}

const editStock = (stock) => {
  editingStock.value = stock
  Object.assign(stockForm, stock)
  showAddModal.value = true
}

const syncData = async (stock) => {
  alert(`正在同步 ${stock.name} 数据...`)
}

const deleteStock = async (stock) => {
  if (confirm(`确定要删除股票 ${stock.name} 吗？`)) {
    stocks.value = stocks.value.filter(s => s.id !== stock.id)
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
  editingStock.value = null
  Object.assign(stockForm, {
    code: '',
    name: '',
    market: '',
    industry: '',
    listDate: '',
    description: '',
    status: 'trading'
  })
}

onMounted(() => {
  loadStocks()
})
</script>
