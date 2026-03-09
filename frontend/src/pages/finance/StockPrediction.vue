<template>
  <div class="space-y-6">
    <PageHeader title="股票预测" subtitle="使用机器学习模型预测股票走势" />

    <Breadcrumb :items="[
      { label: '金融分析' },
      { label: '股票预测' }
    ]" />

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- 左侧：预测配置 -->
      <Card title="预测配置" class="lg:col-span-1">
        <FormSection>
          <Select
            v-model="predictForm.stockId"
            label="选择股票"
            :options="stockOptions"
            placeholder="请选择股票"
            required
          />
          <Select
            v-model="predictForm.model"
            label="预测模型"
            :options="modelOptions"
          />
          <Input
            v-model="predictForm.days"
            type="number"
            label="预测天数"
            hint="预测未来N天的走势"
          />
          <DatePicker
            v-model="predictForm.startDate"
            label="开始日期"
          />
          <DatePicker
            v-model="predictForm.endDate"
            label="结束日期"
          />
        </FormSection>

        <template #footer>
          <Button variant="primary" block :loading="predicting" @click="runPrediction">
            开始预测
          </Button>
        </template>
      </Card>

      <!-- 右侧：预测结果 -->
      <div class="lg:col-span-2 space-y-6">
        <!-- 预测结果概览 -->
        <Card v-if="predictionResult" title="预测结果">
          <div class="grid grid-cols-4 gap-4 mb-6">
            <div class="text-center">
              <p class="text-sm text-secondary-500">预测价格</p>
              <p class="text-2xl font-bold" :class="predictionResult.trend === 'up' ? 'text-success-600' : 'text-danger-600'">
                {{ predictionResult.predictedPrice.toFixed(2) }}
              </p>
            </div>
            <div class="text-center">
              <p class="text-sm text-secondary-500">预测涨跌</p>
              <p class="text-2xl font-bold" :class="predictionResult.change >= 0 ? 'text-success-600' : 'text-danger-600'">
                {{ predictionResult.change >= 0 ? '+' : '' }}{{ predictionResult.change.toFixed(2) }}%
              </p>
            </div>
            <div class="text-center">
              <p class="text-sm text-secondary-500">置信度</p>
              <p class="text-2xl font-bold text-primary-600">{{ predictionResult.confidence }}%</p>
            </div>
            <div class="text-center">
              <p class="text-sm text-secondary-500">预测趋势</p>
              <Badge :variant="predictionResult.trend === 'up' ? 'success' : 'danger'" size="lg">
                {{ predictionResult.trend === 'up' ? '看涨' : '看跌' }}
              </Badge>
            </div>
          </div>

          <!-- 预测图表 -->
          <div class="h-80 bg-secondary-50 rounded-lg flex items-center justify-center text-secondary-400">
            <div class="text-center">
              <svg class="w-16 h-16 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
              <p>预测趋势图表</p>
              <p class="text-sm mt-1">集成 Chart.js 后显示</p>
            </div>
          </div>
        </Card>

        <!-- 无预测结果提示 -->
        <Card v-else title="预测结果">
          <div class="h-80 flex items-center justify-center text-secondary-400">
            <div class="text-center">
              <svg class="w-16 h-16 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <p>请选择股票并开始预测</p>
            </div>
          </div>
        </Card>

        <!-- 历史预测记录 -->
        <Card title="历史预测记录">
          <Table
            :columns="historyColumns"
            :data="historyPredictions"
            :show-pagination="false"
          >
            <template #cell-trend="{ row }">
              <Badge :variant="row.trend === 'up' ? 'success' : 'danger'">
                {{ row.trend === 'up' ? '看涨' : '看跌' }}
              </Badge>
            </template>
            <template #cell-accuracy="{ row }">
              <Progress :value="row.accuracy" :color="row.accuracy >= 70 ? 'success' : 'warning'" />
            </template>
            <template #cell-createdAt="{ row }">
              {{ formatDate(row.createdAt) }}
            </template>
          </Table>
        </Card>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Input from '@/components/ui/Input.vue'
import Select from '@/components/ui/Select.vue'
import DatePicker from '@/components/ui/DatePicker.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Progress from '@/components/ui/Progress.vue'
import FormSection from '@/components/ui/FormSection.vue'

const predicting = ref(false)
const predictionResult = ref(null)

const predictForm = reactive({
  stockId: '',
  model: 'lstm',
  days: 7,
  startDate: '',
  endDate: ''
})

const stockOptions = [
  { value: '1', label: '平安银行 (000001)' },
  { value: '2', label: '万科A (000002)' },
  { value: '3', label: '浦发银行 (600000)' }
]

const modelOptions = [
  { value: 'lstm', label: 'LSTM神经网络' },
  { value: 'gru', label: 'GRU神经网络' },
  { value: 'transformer', label: 'Transformer' },
  { value: 'prophet', label: 'Prophet' }
]

const historyColumns = [
  { key: 'stockName', label: '股票' },
  { key: 'model', label: '模型' },
  { key: 'predictedPrice', label: '预测价格' },
  { key: 'actualPrice', label: '实际价格' },
  { key: 'trend', label: '趋势' },
  { key: 'accuracy', label: '准确率' },
  { key: 'createdAt', label: '预测时间' }
]

const historyPredictions = ref([
  { id: 1, stockName: '平安银行', model: 'LSTM', predictedPrice: 12.50, actualPrice: 12.35, trend: 'up', accuracy: 85, createdAt: '2024-01-14 10:00:00' },
  { id: 2, stockName: '万科A', model: 'GRU', predictedPrice: 8.40, actualPrice: 8.56, trend: 'down', accuracy: 72, createdAt: '2024-01-13 14:30:00' },
  { id: 3, stockName: '浦发银行', model: 'Transformer', predictedPrice: 7.95, actualPrice: 7.89, trend: 'up', accuracy: 91, createdAt: '2024-01-12 09:15:00' }
])

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const runPrediction = async () => {
  predicting.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 2000))
    predictionResult.value = {
      predictedPrice: 12.85,
      change: 4.05,
      confidence: 78,
      trend: 'up'
    }
  } finally {
    predicting.value = false
  }
}
</script>
