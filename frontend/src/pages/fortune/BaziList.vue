<template>
  <div class="space-y-6">
    <PageHeader title="八字数据" subtitle="八字命理分析" />

    <Breadcrumb :items="[{ label: '看相算命' }, { label: '八字数据' }]" />

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- 八字排盘 -->
      <Card title="八字排盘">
        <FormSection>
          <Input v-model="form.name" label="姓名" placeholder="请输入姓名" />
          <RadioCard v-model="form.gender" label="性别" :options="genderOptions" />
          <DatePicker v-model="form.birthDate" label="出生日期" />
          <Input v-model="form.birthTime" label="出生时辰" placeholder="如：子时" />
        </FormSection>
        <template #footer>
          <Button variant="primary" block :loading="calculating" @click="calculateBazi">开始排盘</Button>
        </template>
      </Card>

      <!-- 排盘结果 -->
      <Card v-if="baziResult" title="排盘结果">
        <div class="space-y-4">
          <div class="grid grid-cols-4 gap-2 text-center">
            <div class="p-3 bg-primary-50 rounded-lg">
              <p class="text-xs text-secondary-500">年柱</p>
              <p class="text-xl font-bold text-primary-600">{{ baziResult.year }}</p>
            </div>
            <div class="p-3 bg-success-50 rounded-lg">
              <p class="text-xs text-secondary-500">月柱</p>
              <p class="text-xl font-bold text-success-600">{{ baziResult.month }}</p>
            </div>
            <div class="p-3 bg-warning-50 rounded-lg">
              <p class="text-xs text-secondary-500">日柱</p>
              <p class="text-xl font-bold text-warning-600">{{ baziResult.day }}</p>
            </div>
            <div class="p-3 bg-danger-50 rounded-lg">
              <p class="text-xs text-secondary-500">时柱</p>
              <p class="text-xl font-bold text-danger-600">{{ baziResult.hour }}</p>
            </div>
          </div>
          <div class="p-4 bg-secondary-50 rounded-lg">
            <h4 class="font-semibold mb-2">五行分析</h4>
            <div class="flex flex-wrap gap-2">
              <Badge v-for="(count, element) in baziResult.elements" :key="element" variant="primary">{{ element }}: {{ count }}</Badge>
            </div>
          </div>
          <div class="p-4 bg-secondary-50 rounded-lg">
            <h4 class="font-semibold mb-2">命理分析</h4>
            <p class="text-sm text-secondary-600">{{ baziResult.analysis }}</p>
          </div>
        </div>
      </Card>
    </div>

    <!-- 历史记录 -->
    <Card title="历史排盘记录">
      <Table :columns="historyColumns" :data="historyRecords" :show-pagination="false">
        <template #cell-createdAt="{ row }">
          {{ formatDate(row.createdAt) }}
        </template>
      </Table>
    </Card>
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
import RadioCard from '@/components/ui/RadioCard.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import FormSection from '@/components/ui/FormSection.vue'

const calculating = ref(false)
const baziResult = ref(null)

const form = reactive({ name: '', gender: 'male', birthDate: '', birthTime: '' })
const genderOptions = [{ value: 'male', label: '男' }, { value: 'female', label: '女' }]

const historyColumns = [
  { key: 'name', label: '姓名' },
  { key: 'gender', label: '性别' },
  { key: 'birthDate', label: '出生日期' },
  { key: 'bazi', label: '八字' },
  { key: 'createdAt', label: '排盘时间' }
]

const historyRecords = ref([
  { id: 1, name: '张三', gender: '男', birthDate: '1990-01-15', bazi: '庚午 戊寅 甲子 甲子', createdAt: '2024-01-15' }
])

const formatDate = (date) => date ? new Date(date).toLocaleDateString('zh-CN') : '-'

const calculateBazi = async () => {
  calculating.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 1500))
    baziResult.value = {
      year: '庚午',
      month: '戊寅',
      day: '甲子',
      hour: '甲子',
      elements: { 金: 2, 木: 3, 水: 2, 火: 1, 土: 2 },
      analysis: '此命五行木旺，日主天干为木，生于春季。木旺得金，方成栋梁。'
    }
  } finally {
    calculating.value = false
  }
}
</script>
