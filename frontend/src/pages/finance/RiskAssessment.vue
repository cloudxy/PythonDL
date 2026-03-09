<template>
  <div class="space-y-6">
    <PageHeader title="风险评估" subtitle="股票投资风险评估分析" />

    <Breadcrumb :items="[
      { label: '金融分析' },
      { label: '风险评估' }
    ]" />

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- 左侧：评估配置 -->
      <Card title="风险评估配置" class="lg:col-span-1">
        <FormSection>
          <Select
            v-model="assessForm.stockId"
            label="选择股票"
            :options="stockOptions"
            placeholder="请选择股票"
            required
          />
          <Select
            v-model="assessForm.method"
            label="评估方法"
            :options="methodOptions"
          />
          <Input
            v-model="assessForm.investment"
            type="number"
            label="投资金额"
            hint="用于计算风险敞口"
          />
        </FormSection>

        <template #footer>
          <Button variant="primary" block :loading="assessing" @click="runAssessment">
            开始评估
          </Button>
        </template>
      </Card>

      <!-- 右侧：评估结果 -->
      <div class="lg:col-span-2 space-y-6">
        <!-- 风险概览 -->
        <Card v-if="assessmentResult" title="风险概览">
          <div class="grid grid-cols-4 gap-4 mb-6">
            <div class="text-center">
              <CircleProgress :value="assessmentResult.riskScore" :max="100" color="danger" />
              <p class="text-sm text-secondary-500 mt-2">风险评分</p>
            </div>
            <div class="text-center">
              <CircleProgress :value="assessmentResult.volatility" :max="100" color="warning" />
              <p class="text-sm text-secondary-500 mt-2">波动率</p>
            </div>
            <div class="text-center">
              <CircleProgress :value="assessmentResult.liquidity" :max="100" color="primary" />
              <p class="text-sm text-secondary-500 mt-2">流动性</p>
            </div>
            <div class="text-center">
              <CircleProgress :value="assessmentResult.marketRisk" :max="100" color="danger" />
              <p class="text-sm text-secondary-500 mt-2">市场风险</p>
            </div>
          </div>

          <div class="grid grid-cols-2 gap-6">
            <div>
              <h4 class="font-semibold mb-3">风险等级</h4>
              <div class="flex items-center gap-3">
                <Badge :variant="riskLevelVariant" size="lg">{{ assessmentResult.riskLevel }}</Badge>
                <span class="text-secondary-500">{{ assessmentResult.riskDescription }}</span>
              </div>
            </div>
            <div>
              <h4 class="font-semibold mb-3">投资建议</h4>
              <p class="text-secondary-600">{{ assessmentResult.suggestion }}</p>
            </div>
          </div>
        </Card>

        <!-- 无评估结果提示 -->
        <Card v-else title="风险评估结果">
          <div class="h-64 flex items-center justify-center text-secondary-400">
            <div class="text-center">
              <svg class="w-16 h-16 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              <p>请选择股票并开始评估</p>
            </div>
          </div>
        </Card>

        <!-- 详细风险指标 -->
        <Card v-if="assessmentResult" title="详细风险指标">
          <div class="space-y-4">
            <ProgressBar label="VaR (95%)" :value="assessmentResult.var95" :max="100" color="danger" />
            <ProgressBar label="最大回撤" :value="assessmentResult.maxDrawdown" :max="100" color="warning" />
            <ProgressBar label="夏普比率" :value="assessmentResult.sharpeRatio * 20" :max="100" color="success" />
            <ProgressBar label="Beta系数" :value="assessmentResult.beta * 50" :max="100" color="primary" />
          </div>
        </Card>

        <!-- 历史评估记录 -->
        <Card title="历史评估记录">
          <Table
            :columns="historyColumns"
            :data="historyAssessments"
            :show-pagination="false"
          >
            <template #cell-riskLevel="{ row }">
              <Badge :variant="getRiskVariant(row.riskLevel)">{{ row.riskLevel }}</Badge>
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
import { ref, reactive, computed } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Input from '@/components/ui/Input.vue'
import Select from '@/components/ui/Select.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import ProgressBar from '@/components/ui/ProgressBar.vue'
import CircleProgress from '@/components/ui/CircleProgress.vue'
import FormSection from '@/components/ui/FormSection.vue'

const assessing = ref(false)
const assessmentResult = ref(null)

const assessForm = reactive({
  stockId: '',
  method: 'monte_carlo',
  investment: 100000
})

const stockOptions = [
  { value: '1', label: '平安银行 (000001)' },
  { value: '2', label: '万科A (000002)' },
  { value: '3', label: '浦发银行 (600000)' }
]

const methodOptions = [
  { value: 'monte_carlo', label: '蒙特卡洛模拟' },
  { value: 'var', label: 'VaR风险价值' },
  { value: 'cvar', label: 'CVaR条件风险价值' },
  { value: 'stress_test', label: '压力测试' }
]

const historyColumns = [
  { key: 'stockName', label: '股票' },
  { key: 'method', label: '评估方法' },
  { key: 'riskScore', label: '风险评分' },
  { key: 'riskLevel', label: '风险等级' },
  { key: 'createdAt', label: '评估时间' }
]

const historyAssessments = ref([
  { id: 1, stockName: '平安银行', method: '蒙特卡洛模拟', riskScore: 35, riskLevel: '低风险', createdAt: '2024-01-14 10:00:00' },
  { id: 2, stockName: '万科A', method: 'VaR风险价值', riskScore: 65, riskLevel: '中风险', createdAt: '2024-01-13 14:30:00' },
  { id: 3, stockName: '浦发银行', method: '压力测试', riskScore: 45, riskLevel: '低风险', createdAt: '2024-01-12 09:15:00' }
])

const riskLevelVariant = computed(() => {
  if (!assessmentResult.value) return 'secondary'
  const level = assessmentResult.value.riskLevel
  if (level === '高风险') return 'danger'
  if (level === '中风险') return 'warning'
  return 'success'
})

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const getRiskVariant = (level) => {
  if (level === '高风险') return 'danger'
  if (level === '中风险') return 'warning'
  return 'success'
}

const runAssessment = async () => {
  assessing.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 2000))
    assessmentResult.value = {
      riskScore: 45,
      volatility: 35,
      liquidity: 80,
      marketRisk: 40,
      riskLevel: '低风险',
      riskDescription: '该股票风险较低，适合稳健型投资者',
      suggestion: '建议适度配置，注意分散投资风险',
      var95: 15,
      maxDrawdown: 25,
      sharpeRatio: 1.8,
      beta: 0.85
    }
  } finally {
    assessing.value = false
  }
}
</script>
