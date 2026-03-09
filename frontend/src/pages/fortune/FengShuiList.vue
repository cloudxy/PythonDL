<template>
  <div class="space-y-6">
    <PageHeader title="风水数据管理" subtitle="管理风水分析数据">
      <template #actions>
        <Button variant="primary" @click="showAddModal = true">新增数据</Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[{ label: '看相算命' }, { label: '风水数据' }]" />

    <Card>
      <Table :columns="columns" :data="fengShuiData" :loading="loading">
        <template #cell-direction="{ row }">
          <Badge variant="primary">{{ row.direction }}</Badge>
        </template>
        <template #cell-rating="{ row }">
          <Rating :model-value="row.rating" readonly />
        </template>
        <template #actions="{ row }">
          <Button variant="ghost" size="sm" @click="editItem(row)">编辑</Button>
          <Button variant="danger" size="sm" @click="deleteItem(row)">删除</Button>
        </template>
      </Table>
    </Card>

    <Modal v-model="showAddModal" :title="editingItem ? '编辑数据' : '新增数据'" size="lg" :show-default-footer="true" @confirm="handleSubmit">
      <FormSection>
        <Input v-model="form.name" label="名称" required />
        <Select v-model="form.direction" label="方位" :options="directionOptions" />
        <Input v-model="form.element" label="五行" />
        <Rating v-model="form.rating" label="评分" />
        <Textarea v-model="form.description" label="描述" />
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
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import Rating from '@/components/ui/Rating.vue'

const loading = ref(false)
const showAddModal = ref(false)
const editingItem = ref(null)

const columns = [
  { key: 'id', label: 'ID' },
  { key: 'name', label: '名称' },
  { key: 'direction', label: '方位' },
  { key: 'element', label: '五行' },
  { key: 'rating', label: '评分' },
  { key: 'description', label: '描述' }
]

const fengShuiData = ref([
  { id: 1, name: '东方青龙位', direction: '东', element: '木', rating: 4, description: '代表生机与发展' },
  { id: 2, name: '南方朱雀位', direction: '南', element: '火', rating: 5, description: '代表热情与活力' },
  { id: 3, name: '西方白虎位', direction: '西', element: '金', rating: 3, description: '代表收获与肃杀' }
])

const form = reactive({ name: '', direction: '', element: '', rating: 3, description: '' })

const directionOptions = [
  { value: '东', label: '东方' },
  { value: '南', label: '南方' },
  { value: '西', label: '西方' },
  { value: '北', label: '北方' }
]

const editItem = (item) => { editingItem.value = item; Object.assign(form, item); showAddModal.value = true }
const deleteItem = (item) => { fengShuiData.value = fengShuiData.value.filter(d => d.id !== item.id) }
const handleSubmit = () => { showAddModal.value = false }
</script>
