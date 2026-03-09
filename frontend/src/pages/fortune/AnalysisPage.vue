<template>
  <div class="space-y-6">
    <PageHeader title="综合分析" subtitle="命理综合分析" />

    <Breadcrumb :items="[{ label: '看相算命' }, { label: '综合分析' }]" />

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- 输入表单 -->
      <Card title="分析配置" class="lg:col-span-1">
        <FormSection>
          <Input v-model="form.name" label="姓名" required />
          <RadioCard v-model="form.gender" label="性别" :options="genderOptions" />
          <DatePicker v-model="form.birthDate" label="出生日期" />
          <Input v-model="form.birthTime" label="出生时辰" />
          <Select v-model="form.analysisType" label="分析类型" :options="analysisOptions" />
        </FormSection>
        <template #footer>
          <Button variant="primary" block :loading="analyzing" @click="runAnalysis">开始分析</Button>
        </template>
      </Card>

      <!-- 分析结果 -->
      <div class="lg:col-span-2 space-y-6">
        <Card v-if="analysisResult" title="分析结果">
          <Tabs v-model="activeTab" :tabs="resultTabs" />
          
          <div v-show="activeTab === 'overview'" class="mt-6 space-y-4">
            <div class="grid grid-cols-4 gap-4">
              <div class="text-center p-4 bg-primary-50 rounded-lg">
                <p class="text-sm text-secondary-500">综合评分</p>
                <p class="text-3xl font-bold text-primary-600">{{ analysisResult.score }}</p>
              </div>
              <div class="text-center p-4 bg-success-50 rounded-lg">
                <p class="text-sm text-secondary-500">事业运</p>
                <p class="text-3xl font-bold text-success-600">{{ analysisResult.career }}</p>
              </div>
              <div class="text-center p-4 bg-warning-50 rounded-lg">
                <p class="text-sm text-secondary-500">财运</p>
                <p class="text-3xl font-bold text-warning-600">{{ analysisResult.wealth }}</p>
              </div>
              <div class="text-center p-4 bg-danger-50 rounded-lg">
                <p class="text-sm text-secondary-500">感情运</p>
                <p class="text-3xl font-bold text-danger-600">{{ analysisResult.love }}</p>
              </div>
            </div>
            <div>
              <h4 class="font-semibold mb-2">综合评语</h4>
              <p class="text-secondary-600">{{ analysisResult.summary }}</p>
            </div>
          </div>

          <div v-show="activeTab === 'career'" class="mt-6">
            <h4 class="font-semibold mb-3">事业分析</h4>
            <p class="text-secondary-600">{{ analysisResult.careerAnalysis }}</p>
          </div>

          <div v-show="activeTab === 'wealth'" class="mt-6">
            <h4 class="font-semibold mb-3">财运分析</h4>
            <p class="text-secondary-600">{{ analysisResult.wealthAnalysis }}</p>
          </div>

          <div v-show="activeTab === 'love'" class="mt-6">
            <h4 class="font-semibold mb-3">感情分析</h4>
            <p class="text-secondary-600">{{ analysisResult.loveAnalysis }}</p>
          </div>
        </Card>

        <Card v-else title="分析结果">
          <div class="h-64 flex items-center justify-center text-secondary-400">
            <p>请填写信息并开始分析</p>
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
import DatePicker from '@/components/ui/DatePicker.vue'
import Select from '@/components/ui/Select.vue'
import RadioCard from '@/components/ui/RadioCard.vue'
import Tabs from '@/components/ui/Tabs.vue'
import FormSection from '@/components/ui/FormSection.vue'

const analyzing = ref(false)
const analysisResult = ref(null)
const activeTab = ref('overview')

const form = reactive({ name: '', gender: 'male', birthDate: '', birthTime: '', analysisType: 'comprehensive' })
const genderOptions = [{ value: 'male', label: '男' }, { value: 'female', label: '女' }]
const analysisOptions = [{ value: 'comprehensive', label: '综合分析' }, { value: 'career', label: '事业分析' }, { value: 'wealth', label: '财运分析' }]
const resultTabs = [{ value: 'overview', label: '综合概览' }, { value: 'career', label: '事业' }, { value: 'wealth', label: '财运' }, { value: 'love', label: '感情' }]

const runAnalysis = async () => {
  analyzing.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 2000))
    analysisResult.value = {
      score: 85,
      career: 90,
      wealth: 80,
      love: 85,
      summary: '整体运势良好，事业有成的机会较大，财运稳定，感情生活和谐。建议把握机会，稳中求进。',
      careerAnalysis: '事业方面有贵人相助，适合开展新项目或寻求晋升机会。注意与同事的沟通协作。',
      wealthAnalysis: '财运平稳，正财收入稳定，偏财运一般。建议稳健投资，避免高风险操作。',
      loveAnalysis: '感情运势良好，单身者有机会遇到心仪对象，已婚者家庭和睦。'
    }
  } finally {
    analyzing.value = false
  }
}
</script>
