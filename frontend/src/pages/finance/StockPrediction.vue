<template>
  <div class="space-y-6">
    <PageHeader title="股票预测" subtitle="基于 AI 的股票价格预测">
      <template #actions>
        <Button variant="primary" @click="showCreateModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          新建预测
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '金融分析' },
      { label: '股票预测' }
    ]" />

    <Card>
      <div class="flex items-center justify-between mb-4">
        <SearchBar
          v-model="searchQuery"
          placeholder="搜索预测名称、股票代码..."
          @search="handleSearch"
          @add="showCreateModal = true"
        />
      </div>

      <div v-if="loading" class="flex justify-center py-12">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>

      <div v-else-if="predictions.length === 0" class="text-center py-12">
        <svg class="w-12 h-12 text-secondary-300 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        <p class="text-secondary-500">暂无预测数据</p>
      </div>

      <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div
          v-for="item in predictions"
          :key="item.id"
          class="border border-secondary-200 rounded-lg p-4 hover:shadow-lg transition-shadow"
        >
          <div class="flex items-start justify-between mb-3">
            <div>
              <h3 class="font-semibold text-secondary-900">{{ item.name }}</h3>
              <p class="text-sm text-secondary-500">{{ item.stockCode }} · {{ item.stockName }}</p>
            </div>
            <Badge :variant="getStatusBadgeVariant(item.status)">
              {{ statusMap[item.status] }}
            </Badge>
          </div>
          
          <div class="space-y-2 mb-4">
            <div class="flex items-center justify-between text-sm">
              <span class="text-secondary-500">当前价格</span>
              <span class="font-medium">{{ item.currentPrice }}</span>
            </div>
            <div class="flex items-center justify-between text-sm">
              <span class="text-secondary-500">预测价格</span>
              <span class="font-medium text-primary-600">{{ item.predictedPrice }}</span>
            </div>
            <div class="flex items-center justify-between text-sm">
              <span class="text-secondary-500">预期涨幅</span>
              <span :class="item.expectedChange >= 0 ? 'text-success-600' : 'text-danger-600'">
                {{ item.expectedChange >= 0 ? '+' : '' }}{{ item.expectedChange }}%
              </span>
            </div>
            <div class="flex items-center justify-between text-sm">
              <span class="text-secondary-500">置信度</span>
              <span class="font-medium">{{ item.confidence }}%</span>
            </div>
          </div>

          <div class="flex items-center justify-between text-xs text-secondary-400 mb-4">
            <span>预测周期：{{ item.period }}天</span>
            <span>{{ formatDate(item.createdAt) }}</span>
          </div>

          <div class="flex items-center gap-2">
            <Button
              v-if="item.status === 'pending'"
              variant="primary"
              size="sm"
              class="flex-1"
              @click="runPrediction(item)"
            >
              运行预测
            </Button>
            <Button
              v-else-if="item.status === 'completed'"
              variant="outline"
              size="sm"
              class="flex-1"
              @click="viewResults(item)"
            >
              查看结果
            </Button>
            <Button v-else variant="outline" size="sm" class="flex-1" disabled>
              运行中
            </Button>
            <Button variant="ghost" size="sm" @click="deletePrediction(item)">删除</Button>
          </div>
        </div>
      </div>
    </Card>

    <!-- 新建预测模态框 -->
    <Modal
      v-model="showCreateModal"
      title="新建股票预测"
      size="lg"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <FormSection title="预测配置">
        <Input
          v-model="predictionForm.name"
          label="预测名称"
          placeholder="请输入预测名称"
          :error="formErrors.name"
          required
        />
        <Select
          v-model="predictionForm.stockCode"
          label="选择股票"
          :options="stockOptions"
          placeholder="请选择股票"
          :error="formErrors.stockCode"
          required
        />
        <Select
          v-model="predictionForm.period"
          label="预测周期"
          :options="periodOptions"
          placeholder="请选择预测周期"
          required
        />
        <Select
          v-model="predictionForm.model"
          label="预测模型"
          :options="modelOptions"
          placeholder="请选择预测模型"
          required
        />
      </FormSection>
    </Modal>

    <!-- 预测结果模态框 -->
    <Modal
      v-model="showResultModal"
      title="预测结果"
      size="xl"
    >
      <div v-if="currentPrediction" class="space-y-6">
        <div class="grid grid-cols-3 gap-4">
          <div class="text-center p-4 bg-secondary-50 rounded-lg">
            <p class="text-sm text-secondary-500 mb-1">当前价格</p>
            <p class="text-2xl font-bold text-secondary-900">{{ currentPrediction.currentPrice }}</p>
          </div>
          <div class="text-center p-4 bg-primary-50 rounded-lg">
            <p class="text-sm text-primary-600 mb-1">预测价格</p>
            <p class="text-2xl font-bold text-primary-600">{{ currentPrediction.predictedPrice }}</p>
          </div>
          <div class="text-center p-4 bg-success-50 rounded-lg">
            <p class="text-sm text-success-600 mb-1">预期涨幅</p>
            <p class="text-2xl font-bold text-success-600">
              {{ currentPrediction.expectedChange >= 0 ? '+' : '' }}{{ currentPrediction.expectedChange }}%
            </p>
          </div>
        </div>

        <div class="h-64 bg-secondary-50 rounded-lg flex items-center justify-center text-secondary-400">
          <p>价格走势图（集成图表库后显示）</p>
        </div>

        <div class="space-y-3">
          <h4 class="font-semibold">预测详情</h4>
          <div class="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span class="text-secondary-500">置信度：</span>
              <span class="font-medium">{{ currentPrediction.confidence }}%</span>
            </div>
            <div>
              <span class="text-secondary-500">预测周期：</span>
              <span class="font-medium">{{ currentPrediction.period }} 天</span>
            </div>
            <div>
              <span class="text-secondary-500">使用模型：</span>
              <span class="font-medium">{{ currentPrediction.model }}</span>
            </div>
            <div>
              <span class="text-secondary-500">完成时间：</span>
              <span class="font-medium">{{ formatDate(currentPrediction.completedAt) }}</span>
            </div>
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
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import SearchBar from '@/components/ui/SearchBar.vue'

const loading = ref(false)
const submitting = ref(false)
const showCreateModal = ref(false)
const showResultModal = ref(false)
const searchQuery = ref('')

const statusMap = {
  pending: '待运行',
  running: '运行中',
  completed: '已完成',
  failed: '失败'
}

const predictions = ref([
  { id: 1, name: '平安银行预测', stockCode: '000001', stockName: '平安银行', currentPrice: 12.35, predictedPrice: 13.20, expectedChange: 6.88, confidence: 85, period: 7, status: 'completed', createdAt: '2024-01-15', completedAt: '2024-01-15 10:30:00' },
  { id: 2, name: '万科 A 预测', stockCode: '000002', stockName: '万科 A', currentPrice: 8.56, predictedPrice: 8.90, expectedChange: 3.97, confidence: 78, period: 7, status: 'pending', createdAt: '2024-01-14' },
  { id: 3, name: '浦发银行预测', stockCode: '600000', stockName: '浦发银行', currentPrice: 7.89, predictedPrice: 7.65, expectedChange: -3.04, confidence: 82, period: 14, status: 'running', createdAt: '2024-01-14' }
])

const predictionForm = reactive({
  name: '',
  stockCode: '',
  period: '7',
  model: 'lstm'
})

const formErrors = reactive({
  name: '',
  stockCode: ''
})

const stockOptions = [
  { value: '000001', label: '000001 - 平安银行' },
  { value: '000002', label: '000002 - 万科 A' },
  { value: '600000', label: '600000 - 浦发银行' }
]

const periodOptions = [
  { value: '7', label: '7 天' },
  { value: '14', label: '14 天' },
  { value: '30', label: '30 天' }
]

const modelOptions = [
  { value: 'lstm', label: 'LSTM 长短期记忆网络' },
  { value: 'gru', label: 'GRU 门控循环单元' },
  { value: 'transformer', label: 'Transformer 模型' }
]

const currentPrediction = ref(null)

const getStatusBadgeVariant = (status) => {
  const variants = {
    pending: 'secondary',
    running: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return variants[status] || 'secondary'
}

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const handleSearch = () => {
  loadPredictions()
}

const loadPredictions = async () => {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
  } finally {
    loading.value = false
  }
}

const runPrediction = async (item) => {
  if (confirm(`确定要运行 ${item.name} 吗？`)) {
    item.status = 'running'
    setTimeout(() => {
      item.status = 'completed'
      item.completedAt = new Date().toISOString()
    }, 3000)
  }
}

const viewResults = (item) => {
  currentPrediction.value = item
  showResultModal.value = true
}

const deletePrediction = async (item) => {
  if (confirm(`确定要删除 ${item.name} 吗？`)) {
    predictions.value = predictions.value.filter(p => p.id !== item.id)
  }
}

const handleSubmit = async () => {
  submitting.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 1000))
    const stock = stockOptions.find(s => s.value === predictionForm.stockCode)
    predictions.value.unshift({
      id: Date.now(),
      name: predictionForm.name,
      stockCode: predictionForm.stockCode,
      stockName: stock?.label.split(' - ')[1] || '',
      currentPrice: 0,
      predictedPrice: 0,
      expectedChange: 0,
      confidence: 0,
      period: parseInt(predictionForm.period),
      status: 'pending',
      createdAt: new Date().toISOString().split('T')[0]
    })
    showCreateModal.value = false
    resetForm()
  } finally {
    submitting.value = false
  }
}

const resetForm = () => {
  Object.assign(predictionForm, {
    name: '',
    stockCode: '',
    period: '7',
    model: 'lstm'
  })
  Object.assign(formErrors, {
    name: '',
    stockCode: ''
  })
}

onMounted(() => {
  loadPredictions()
})
</script>
