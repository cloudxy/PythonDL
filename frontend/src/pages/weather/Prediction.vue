<template>
  <div class="space-y-6">
    <PageHeader title="气象预测" subtitle="基于历史数据进行气象预测" />

    <Breadcrumb :items="[
      { label: '气象分析' },
      { label: '气象预测' }
    ]" />

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- 预测配置 -->
      <Card title="预测配置" class="lg:col-span-1">
        <FormSection>
          <Select v-model="predictForm.stationId" label="选择站点" :options="stationOptions" required />
          <Select v-model="predictForm.model" label="预测模型" :options="modelOptions" />
          <Input v-model="predictForm.days" type="number" label="预测天数" />
        </FormSection>
        <template #footer>
          <Button variant="primary" block :loading="predicting" @click="runPrediction">开始预测</Button>
        </template>
      </Card>

      <!-- 预测结果 -->
      <div class="lg:col-span-2 space-y-6">
        <Card title="预测结果">
          <div v-if="predictionResult" class="space-y-6">
            <div class="grid grid-cols-3 gap-4">
              <div class="text-center p-4 bg-secondary-50 rounded-lg">
                <p class="text-sm text-secondary-500">预测最高温</p>
                <p class="text-2xl font-bold text-danger-600">{{ predictionResult.maxTemp }}°C</p>
              </div>
              <div class="text-center p-4 bg-secondary-50 rounded-lg">
                <p class="text-sm text-secondary-500">预测最低温</p>
                <p class="text-2xl font-bold text-primary-600">{{ predictionResult.minTemp }}°C</p>
              </div>
              <div class="text-center p-4 bg-secondary-50 rounded-lg">
                <p class="text-sm text-secondary-500">预测降水概率</p>
                <p class="text-2xl font-bold text-warning-600">{{ predictionResult.rainProb }}%</p>
              </div>
            </div>

            <!-- 预测图表占位 -->
            <div class="h-64 bg-secondary-50 rounded-lg flex items-center justify-center text-secondary-400">
              <p>预测趋势图表</p>
            </div>
          </div>
          <div v-else class="h-64 flex items-center justify-center text-secondary-400">
            <p>请配置参数并开始预测</p>
          </div>
        </Card>

        <!-- 未来7天预报 -->
        <Card title="未来7天预报">
          <div class="grid grid-cols-7 gap-2">
            <div v-for="(day, index) in forecast" :key="index" class="text-center p-3 bg-secondary-50 rounded-lg">
              <p class="text-xs text-secondary-500">{{ day.date }}</p>
              <p class="text-lg font-bold mt-1">{{ day.temp }}°</p>
              <Badge :variant="getWeatherVariant(day.weather)" size="sm" class="mt-1">{{ day.weather }}</Badge>
            </div>
          </div>
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
import Badge from '@/components/ui/Badge.vue'
import FormSection from '@/components/ui/FormSection.vue'

const predicting = ref(false)
const predictionResult = ref(null)

const predictForm = reactive({
  stationId: '',
  model: 'lstm',
  days: 7
})

const stationOptions = [
  { value: '1', label: '北京气象站' },
  { value: '2', label: '上海气象站' }
]

const modelOptions = [
  { value: 'lstm', label: 'LSTM神经网络' },
  { value: 'arima', label: 'ARIMA模型' },
  { value: 'prophet', label: 'Prophet' }
]

const forecast = ref([
  { date: '周一', temp: 8, weather: '晴' },
  { date: '周二', temp: 10, weather: '多云' },
  { date: '周三', temp: 7, weather: '小雨' },
  { date: '周四', temp: 5, weather: '阴' },
  { date: '周五', temp: 6, weather: '晴' },
  { date: '周六', temp: 9, weather: '晴' },
  { date: '周日', temp: 11, weather: '多云' }
])

const getWeatherVariant = (weather) => {
  const variants = { '晴': 'success', '多云': 'primary', '阴': 'secondary', '小雨': 'warning' }
  return variants[weather] || 'secondary'
}

const runPrediction = async () => {
  predicting.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 2000))
    predictionResult.value = {
      maxTemp: 12,
      minTemp: 3,
      rainProb: 35
    }
  } finally {
    predicting.value = false
  }
}
</script>
