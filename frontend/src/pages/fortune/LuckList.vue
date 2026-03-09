<template>
  <div class="space-y-6">
    <PageHeader title="运势数据" subtitle="每日运势数据管理" />

    <Breadcrumb :items="[{ label: '看相算命' }, { label: '运势数据' }]" />

    <Card>
      <div class="flex items-center gap-4 mb-6">
        <DatePicker v-model="selectedDate" label="" />
        <Button variant="primary" @click="loadFortune">查询</Button>
      </div>

      <Table :columns="columns" :data="fortuneData" :loading="loading">
        <template #cell-fortune="{ row }">
          <Badge :variant="getFortuneVariant(row.fortune)">{{ row.fortune }}</Badge>
        </template>
        <template #cell-actions="{ row }">
          <Button variant="ghost" size="sm" @click="editFortune(row)">编辑</Button>
        </template>
      </Table>
    </Card>

    <Modal v-model="showEditModal" title="编辑运势" size="md" :show-default-footer="true" @confirm="handleSave">
      <FormSection v-if="editingFortune">
        <Input v-model="editForm.target" label="对象" disabled />
        <Select v-model="editForm.fortune" label="运势等级" :options="fortuneOptions" />
        <Textarea v-model="editForm.description" label="运势描述" />
        <Textarea v-model="editForm.advice" label="今日建议" />
      </FormSection>
    </Modal>
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
import Textarea from '@/components/ui/Textarea.vue'
import DatePicker from '@/components/ui/DatePicker.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'

const loading = ref(false)
const showEditModal = ref(false)
const editingFortune = ref(null)
const selectedDate = ref(new Date().toISOString().split('T')[0])

const columns = [
  { key: 'id', label: 'ID' },
  { key: 'target', label: '对象' },
  { key: 'fortune', label: '运势' },
  { key: 'description', label: '描述' },
  { key: 'date', label: '日期' }
]

const fortuneData = ref([
  { id: 1, target: '属鼠', fortune: '大吉', description: '今日运势极佳，诸事顺遂', date: '2024-01-15' },
  { id: 2, target: '属牛', fortune: '中吉', description: '平稳发展，稳中求进', date: '2024-01-15' },
  { id: 3, target: '属虎', fortune: '小吉', description: '有小惊喜，注意细节', date: '2024-01-15' }
])

const editForm = reactive({ target: '', fortune: '', description: '', advice: '' })

const fortuneOptions = [
  { value: '大吉', label: '大吉' },
  { value: '中吉', label: '中吉' },
  { value: '小吉', label: '小吉' },
  { value: '平', label: '平' },
  { value: '小凶', label: '小凶' }
]

const getFortuneVariant = (fortune) => {
  const variants = { '大吉': 'success', '中吉': 'primary', '小吉': 'info', '平': 'secondary', '小凶': 'warning' }
  return variants[fortune] || 'secondary'
}

const loadFortune = () => { loading.value = true; setTimeout(() => loading.value = false, 500) }
const editFortune = (item) => { editingFortune.value = item; Object.assign(editForm, item); showEditModal.value = true }
const handleSave = () => { showEditModal.value = false }
</script>
