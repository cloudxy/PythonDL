<template>
  <div class="space-y-6">
    <PageHeader title="人口数据管理" subtitle="管理人口统计数据">
      <template #actions>
        <Button variant="primary" @click="showAddModal = true">新增数据</Button>
      </template>
    </PageHeader>
    <Breadcrumb :items="[{ label: '消费分析' }, { label: '人口数据' }]" />
    <Card>
      <Table :columns="columns" :data="populationData" :loading="loading">
        <template #cell-urbanRate="{ row }">
          <Progress :value="row.urbanRate" color="primary" />
        </template>
        <template #cell-actions="{ row }">
          <Button variant="ghost" size="sm" @click="editItem(row)">编辑</Button>
          <Button variant="danger" size="sm" @click="deleteItem(row)">删除</Button>
        </template>
      </Table>
    </Card>
    <Modal v-model="showAddModal" :title="editingItem ? '编辑数据' : '新增数据'" size="md" :show-default-footer="true" @confirm="handleSubmit">
      <FormSection>
        <Input v-model="form.year" type="number" label="年份" required />
        <Input v-model="form.region" label="地区" required />
        <Input v-model="form.total" type="number" label="总人口(万人)" />
        <Input v-model="form.birthRate" type="number" label="出生率(‰)" />
        <Input v-model="form.urbanRate" type="number" label="城镇化率(%)" />
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
import Table from '@/components/ui/Table.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import Progress from '@/components/ui/Progress.vue'

const loading = ref(false)
const showAddModal = ref(false)
const editingItem = ref(null)

const columns = [
  { key: 'id', label: 'ID' },
  { key: 'year', label: '年份' },
  { key: 'region', label: '地区' },
  { key: 'total', label: '人口(万人)' },
  { key: 'birthRate', label: '出生率(‰)' },
  { key: 'urbanRate', label: '城镇化率' }
]

const populationData = ref([
  { id: 1, year: 2023, region: '全国', total: 140967, birthRate: 6.77, urbanRate: 66.16 },
  { id: 2, year: 2023, region: '北京市', total: 2185, birthRate: 5.67, urbanRate: 87.6 },
  { id: 3, year: 2023, region: '上海市', total: 2487, birthRate: 4.35, urbanRate: 89.3 }
])

const form = reactive({ year: '', region: '', total: '', birthRate: '', urbanRate: '' })

const editItem = (item) => { editingItem.value = item; Object.assign(form, item); showAddModal.value = true }
const deleteItem = (item) => { populationData.value = populationData.value.filter(d => d.id !== item.id) }
const handleSubmit = () => { showAddModal.value = false }
</script>
