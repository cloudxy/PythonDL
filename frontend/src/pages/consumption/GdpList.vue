<template>
  <div class="space-y-6">
    <PageHeader title="GDP数据管理" subtitle="管理GDP经济数据">
      <template #actions>
        <Button variant="primary" @click="showAddModal = true">新增数据</Button>
      </template>
    </PageHeader>
    <Breadcrumb :items="[{ label: '消费分析' }, { label: 'GDP数据' }]" />
    <Card>
      <Table :columns="columns" :data="gdpData" :loading="loading" :total="total" :page="page" :page-size="pageSize" @page-change="handlePageChange">
        <template #cell-growthRate="{ row }">
          <span :class="row.growthRate >= 0 ? 'text-success-600' : 'text-danger-600'">{{ row.growthRate >= 0 ? '+' : '' }}{{ row.growthRate }}%</span>
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
        <Input v-model="form.gdp" type="number" label="GDP总值(亿元)" required />
        <Input v-model="form.growthRate" type="number" label="增长率(%)" />
        <Select v-model="form.industry" label="产业类型" :options="industryOptions" />
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
import Table from '@/components/ui/Table.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'

const loading = ref(false)
const showAddModal = ref(false)
const editingItem = ref(null)
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)

const columns = [
  { key: 'id', label: 'ID' },
  { key: 'year', label: '年份' },
  { key: 'region', label: '地区' },
  { key: 'gdp', label: 'GDP(亿元)' },
  { key: 'growthRate', label: '增长率' },
  { key: 'industry', label: '产业' }
]

const gdpData = ref([
  { id: 1, year: 2023, region: '北京市', gdp: 41610.9, growthRate: 5.2, industry: '第三产业' },
  { id: 2, year: 2023, region: '上海市', gdp: 47218.66, growthRate: 5.0, industry: '第三产业' },
  { id: 3, year: 2023, region: '广东省', gdp: 135673.16, growthRate: 4.8, industry: '第二产业' }
])

const form = reactive({ year: '', region: '', gdp: '', growthRate: '', industry: '' })
const industryOptions = [{ value: 'primary', label: '第一产业' }, { value: 'secondary', label: '第二产业' }, { value: 'tertiary', label: '第三产业' }]

const handlePageChange = ({ page: newPage }) => { page.value = newPage }
const editItem = (item) => { editingItem.value = item; Object.assign(form, item); showAddModal.value = true }
const deleteItem = (item) => { gdpData.value = gdpData.value.filter(d => d.id !== item.id) }
const handleSubmit = () => { showAddModal.value = false }
</script>
