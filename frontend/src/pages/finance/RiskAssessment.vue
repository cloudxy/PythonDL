<template>
  <div class="space-y-6">
    <PageHeader title="风险评估" subtitle="股票投资风险智能评估">
      <template #actions>
        <Button variant="primary" @click="showCreateModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          新建评估
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '金融分析' },
      { label: '风险评估' }
    ]" />

    <Card>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatCard title="总评估数" value="24" icon="📊" trend="+12%" />
        <StatCard title="高风险" value="3" icon="⚠️" trend="-5%" trend-negative />
        <StatCard title="中风险" value="8" icon="⚡" trend="+2%" />
        <StatCard title="低风险" value="13" icon="✅" trend="+8%" />
      </div>

      <Table :columns="columns" :data="assessments" :loading="loading">
        <template #cell-riskLevel="{ row }">
          <Badge :variant="getRiskBadgeVariant(row.riskLevel)">
            {{ riskLevelMap[row.riskLevel] }}
          </Badge>
        </template>
        <template #cell-score="{ row }">
          <div class="flex items-center gap-2">
            <span class="font-medium">{{ row.score }}</span>
            <div class="w-24 h-2 bg-secondary-200 rounded-full overflow-hidden">
              <div
                :class="getScoreColorClass(row.score)"
                class="h-full rounded-full"
                :style="{ width: row.score + '%' }"
              ></div>
            </div>
          </div>
        </template>
        <template #cell-factors="{ row }">
          <div class="flex flex-wrap gap-1">
            <Badge v-for="factor in row.factors" :key="factor" variant="secondary" size="sm">
              {{ factor }}
            </Badge>
          </div>
        </template>
        <template #cell-createdAt="{ row }">
          {{ formatDate(row.createdAt) }}
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button variant="ghost" size="sm" @click="viewReport(row)">报告</Button>
            <Button variant="ghost" size="sm" @click="runAssessment(row)">重评</Button>
            <Button variant="danger" size="sm" @click="deleteAssessment(row)">删除</Button>
          </div>
        </template>
      </Table>
    </Card>

    <!-- 新建评估模态框 -->
    <Modal
      v-model="showCreateModal"
      title="新建风险评估"
      size="lg"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <FormSection title="评估配置">
        <Select
          v-model="assessmentForm.stockCode"
          label="选择股票"
          :options="stockOptions"
          placeholder="请选择股票"
          :error="formErrors.stockCode"
          required
        />
        <Select
          v-model="assessmentForm.type"
          label="评估类型"
          :options="typeOptions"
          placeholder="请选择评估类型"
          required
        />
        <Select
          v-model="assessmentForm.model"
          label="评估模型"
          :options="modelOptions"
          placeholder="请选择评估模型"
          required
        />
      </FormSection>
    </Modal>

    <!-- 评估报告模态框 -->
    <Modal
      v-model="showReportModal"
      title="风险评估报告"
      size="xl"
    >
      <div v-if="currentAssessment" class="space-y-6">
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-xl font-semibold">{{ currentAssessment.stockName }}</h3>
            <p class="text-secondary-500">{{ currentAssessment.stockCode }}</p>
          </div>
          <Badge :variant="getRiskBadgeVariant(currentAssessment.riskLevel)" size="lg">
            {{ riskLevelMap[currentAssessment.riskLevel] }}
          </Badge>
        </div>

        <div class="text-center py-6">
          <div class="inline-block relative">
            <svg class="w-32 h-32 transform -rotate-90">
              <circle
                cx="64"
                cy="64"
                r="56"
                stroke="#e5e7eb"
                stroke-width="16"
                fill="none"
              />
              <circle
                cx="64"
                cy="64"
                r="56"
                :stroke="getScoreColor(currentAssessment.score)"
                stroke-width="16"
                fill="none"
                :stroke-dasharray="351.86"
                :stroke-dashoffset="351.86 * (1 - currentAssessment.score / 100)"
                stroke-linecap="round"
              />
            </svg>
            <div class="absolute inset-0 flex items-center justify-center">
              <span class="text-2xl font-bold">{{ currentAssessment.score }}</span>
            </div>
          </div>
          <p class="mt-2 text-secondary-500">风险评分</p>
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div class="p-4 bg-secondary-50 rounded-lg">
            <h4 class="font-semibold mb-2">市场风险</h4>
            <div class="space-y-1 text-sm">
              <div class="flex justify-between">
                <span class="text-secondary-500">波动率</span>
                <span>{{ currentAssessment.marketVolatility }}%</span>
              </div>
              <div class="flex justify-between">
                <span class="text-secondary-500">Beta 系数</span>
                <span>{{ currentAssessment.beta }}</span>
              </div>
            </div>
          </div>
          <div class="p-4 bg-secondary-50 rounded-lg">
            <h4 class="font-semibold mb-2">财务风险</h4>
            <div class="space-y-1 text-sm">
              <div class="flex justify-between">
                <span class="text-secondary-500">负债率</span>
                <span>{{ currentAssessment.debtRatio }}%</span>
              </div>
              <div class="flex justify-between">
                <span class="text-secondary-500">流动比率</span>
                <span>{{ currentAssessment.currentRatio }}</span>
              </div>
            </div>
          </div>
        </div>

        <div>
          <h4 class="font-semibold mb-2">风险因素</h4>
          <div class="flex flex-wrap gap-2">
            <Badge v-for="factor in currentAssessment.factors" :key="factor" variant="warning">
              {{ factor }}
            </Badge>
          </div>
        </div>

        <div>
          <h4 class="font-semibold mb-2">投资建议</h4>
          <p class="text-secondary-700">{{ currentAssessment.recommendation }}</p>
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
import Select from '@/components/ui/Select.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import StatCard from '@/components/ui/StatCard.vue'

const loading = ref(false)
const submitting = ref(false)
const showCreateModal = ref(false)
const showReportModal = ref(false)

const riskLevelMap = {
  low: '低风险',
  medium: '中风险',
  high: '高风险'
}

const columns = [
  { key: 'stockCode', label: '股票代码' },
  { key: 'stockName', label: '股票名称' },
  { key: 'riskLevel', label: '风险等级' },
  { key: 'score', label: '风险评分' },
  { key: 'factors', label: '风险因素', width: '300px' },
  { key: 'createdAt', label: '评估时间' }
]

const assessments = ref([
  { id: 1, stockCode: '000001', stockName: '平安银行', riskLevel: 'low', score: 25, factors: ['行业风险'], marketVolatility: 18.5, beta: 0.85, debtRatio: 45.2, currentRatio: 1.2, recommendation: '建议持有，风险可控', createdAt: '2024-01-15 10:30:00' },
  { id: 2, stockCode: '000002', stockName: '万科 A', riskLevel: 'medium', score: 55, factors: ['政策风险', '市场风险'], marketVolatility: 25.3, beta: 1.15, debtRatio: 62.8, currentRatio: 0.9, recommendation: '谨慎关注，注意风险', createdAt: '2024-01-15 11:00:00' },
  { id: 3, stockCode: '600000', stockName: '浦发银行', riskLevel: 'high', score: 78, factors: ['信用风险', '流动性风险', '市场风险'], marketVolatility: 32.1, beta: 1.35, debtRatio: 78.5, currentRatio: 0.75, recommendation: '风险较高，建议规避', createdAt: '2024-01-14 15:20:00' }
])

const assessmentForm = reactive({
  stockCode: '',
  type: 'comprehensive',
  model: 'ml'
})

const formErrors = reactive({
  stockCode: ''
})

const stockOptions = [
  { value: '000001', label: '000001 - 平安银行' },
  { value: '000002', label: '000002 - 万科 A' },
  { value: '600000', label: '600000 - 浦发银行' }
]

const typeOptions = [
  { value: 'comprehensive', label: '综合评估' },
  { value: 'market', label: '市场风险' },
  { value: 'financial', label: '财务风险' },
  { value: 'technical', label: '技术风险' }
]

const modelOptions = [
  { value: 'ml', label: '机器学习模型' },
  { value: 'var', label: 'VaR 模型' },
  { value: 'montecarlo', label: '蒙特卡洛模拟' }
]

const currentAssessment = ref(null)

const getRiskBadgeVariant = (level) => {
  const variants = {
    low: 'success',
    medium: 'warning',
    high: 'danger'
  }
  return variants[level] || 'secondary'
}

const getScoreColorClass = (score) => {
  if (score < 40) return 'bg-success-500'
  if (score < 70) return 'bg-warning-500'
  return 'bg-danger-500'
}

const getScoreColor = (score) => {
  if (score < 40) return '#10b981'
  if (score < 70) return '#f59e0b'
  return '#ef4444'
}

const formatDate = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const loadAssessments = async () => {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
  } finally {
    loading.value = false
  }
}

const runAssessment = async (item) => {
  if (confirm(`确定要重新评估 ${item.stockName} 吗？`)) {
    alert('评估已启动，请稍候查看结果')
  }
}

const viewReport = (item) => {
  currentAssessment.value = {
    ...item,
    marketVolatility: 25.3,
    beta: 1.15,
    debtRatio: 62.8,
    currentRatio: 0.9
  }
  showReportModal.value = true
}

const deleteAssessment = async (item) => {
  if (confirm(`确定要删除 ${item.stockName} 的评估记录吗？`)) {
    assessments.value = assessments.value.filter(a => a.id !== item.id)
  }
}

const handleSubmit = async () => {
  submitting.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 1000))
    const stock = stockOptions.find(s => s.value === assessmentForm.stockCode)
    assessments.value.unshift({
      id: Date.now(),
      stockCode: assessmentForm.stockCode,
      stockName: stock?.label.split(' - ')[1] || '',
      riskLevel: 'medium',
      score: 50,
      factors: ['待评估'],
      createdAt: new Date().toISOString()
    })
    showCreateModal.value = false
    resetForm()
  } finally {
    submitting.value = false
  }
}

const resetForm = () => {
  Object.assign(assessmentForm, {
    stockCode: '',
    type: 'comprehensive',
    model: 'ml'
  })
  Object.assign(formErrors, {
    stockCode: ''
  })
}

onMounted(() => {
  loadAssessments()
})
</script>
