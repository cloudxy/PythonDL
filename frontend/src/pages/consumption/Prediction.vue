<template>
  <div class="space-y-6">
    <PageHeader title="消费预测" subtitle="消费趋势预测分析" />
    <Breadcrumb :items="[{ label: '消费分析' }, { label: '消费预测' }]" />

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <Card title="预测配置" class="lg:col-span-1">
        <FormSection>
          <Select v-model="predictForm.region" label="地区" :options="regionOptions" />
          <Select v-model="predictForm.indicator" label="预测指标" :options="indicatorOptions" />
          <Select v-model="predictForm.model" label="预测模型" :options="modelOptions" />
          <Input v-model="predictForm.months" type="number" label="预测月数" />
        </FormSection>
        <template #footer>
          <Button variant="primary" block :loading="predicting" @click="runPrediction">开始预测</Button>
        </template>
      </Card>

      <div class="lg:col-span-2 space-y-6">
        <Card v-if="predictionResult" title="预测结果">
          <div class="grid grid-cols-3 gap-4 mb-6">
            <div class="text-center p-4 bg-primary-50 rounded-lg">
              <p class="text-sm text-secondary-500">预测值</p>
              <p class="text-2xl font-bold text-primary-600">{{ predictionResult.value }}</p>
            </div>
            <div class="text-center p-4 bg-success-50 rounded-lg">
              <p class="text-sm text-secondary-500">增长率</p>
              <p class="text-2xl font-bold text-success-600">+{{ predictionResult.growth }}%</p>
            </div>
            <div class="text-center p-4 bg-warning-50 rounded-lg">
              <p class="text-sm text-secondary-500">置信度</p>
              <p class="text-2xl font-bold text-warning-600">{{ predictionResult.confidence }}%</p>
            </div>
          </div>
          <div class="h-64 bg-secondary-50 rounded-lg flex items-center justify-center text-secondary-400">
            <p>预测趋势图表</p>
          </div>
        </Card>
        <Card v-else title="预测结果">
          <div class="h-64 flex items-center justify-center text-secondary-400"><p>请配置参数并开始预测</p></div>
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
import FormSection from '@/components/ui/FormSection.vue'

const predicting = ref(false)
const predictionResult = ref(null)

const predictForm = reactive({ region: '', indicator: '', model: 'arima', months: 6 })
const regionOptions = [{ value: 'national', label: '全国' }, { value: 'beijing', label: '北京' }, { value: 'shanghai', label: '上海' }]
const indicatorOptions = [{ value: 'consumption', label: '社会消费品零售总额' }, { value: 'cpi', label: 'CPI' }, { value: 'gdp', label: 'GDP' }]
const modelOptions = [{ value: 'arima', label: 'ARIMA' }, { value: 'prophet', label: 'Prophet' }, { value: 'lstm', label: 'LSTM' }]

const runPrediction = async () => {
  predicting.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 2000))
    predictionResult.value = { value: '45,230亿', growth: 7.5, confidence: 85 }
  } finally {
    predicting.value = false
  }
}
</script>
