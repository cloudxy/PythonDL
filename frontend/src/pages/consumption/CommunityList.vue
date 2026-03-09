<template>
  <div class="space-y-6">
    <PageHeader title="小区数据" subtitle="小区消费数据分析" />
    <Breadcrumb :items="[{ label: '消费分析' }, { label: '小区数据' }]" />

    <Card>
      <div class="flex items-center gap-4 mb-6">
        <Input v-model="searchQuery" placeholder="搜索小区名称..." class="flex-1" />
        <Select v-model="filters.city" :options="cityOptions" placeholder="选择城市" />
        <Button variant="primary" @click="handleSearch">搜索</Button>
      </div>

      <Table :columns="columns" :data="communityData" :loading="loading">
        <template #cell-level="{ row }">
          <Badge :variant="row.level === '高端' ? 'success' : row.level === '中端' ? 'primary' : 'secondary'">{{ row.level }}</Badge>
        </template>
        <template #cell-consumption="{ row }">
          {{ row.consumption.toLocaleString() }} 元/月
        </template>
        <template #cell-actions="{ row }">
          <Button variant="ghost" size="sm" @click="viewDetail(row)">详情</Button>
        </template>
      </Table>
    </Card>

    <Modal v-model="showDetailModal" title="小区详情" size="lg">
      <div v-if="currentCommunity" class="space-y-4">
        <div class="grid grid-cols-2 gap-4">
          <div><label class="text-sm text-secondary-500">小区名称</label><p class="font-medium">{{ currentCommunity.name }}</p></div>
          <div><label class="text-sm text-secondary-500">所在城市</label><p class="font-medium">{{ currentCommunity.city }}</p></div>
          <div><label class="text-sm text-secondary-500">户数</label><p class="font-medium">{{ currentCommunity.households }}户</p></div>
          <div><label class="text-sm text-secondary-500">人均消费</label><p class="font-medium">{{ currentCommunity.consumption.toLocaleString() }}元/月</p></div>
          <div><label class="text-sm text-secondary-500">消费等级</label><Badge :variant="currentCommunity.level === '高端' ? 'success' : 'primary'">{{ currentCommunity.level }}</Badge></div>
          <div><label class="text-sm text-secondary-500">建成年份</label><p class="font-medium">{{ currentCommunity.buildYear }}年</p></div>
        </div>
        <div>
          <h4 class="font-semibold mb-2">消费结构分析</h4>
          <div class="space-y-2">
            <ProgressBar label="餐饮" :value="35" color="primary" />
            <ProgressBar label="购物" :value="25" color="success" />
            <ProgressBar label="娱乐" :value="20" color="warning" />
            <ProgressBar label="教育" :value="15" color="danger" />
            <ProgressBar label="其他" :value="5" color="secondary" />
          </div>
        </div>
      </div>
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
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import ProgressBar from '@/components/ui/ProgressBar.vue'

const loading = ref(false)
const searchQuery = ref('')
const showDetailModal = ref(false)
const currentCommunity = ref(null)

const filters = reactive({ city: '' })
const cityOptions = [{ value: 'beijing', label: '北京' }, { value: 'shanghai', label: '上海' }, { value: 'guangzhou', label: '广州' }]

const columns = [
  { key: 'id', label: 'ID' },
  { key: 'name', label: '小区名称' },
  { key: 'city', label: '城市' },
  { key: 'households', label: '户数' },
  { key: 'consumption', label: '人均消费' },
  { key: 'level', label: '消费等级' },
  { key: 'buildYear', label: '建成年份' }
]

const communityData = ref([
  { id: 1, name: '万科城市花园', city: '北京', households: 2500, consumption: 8500, level: '高端', buildYear: 2018 },
  { id: 2, name: '保利香槟国际', city: '上海', households: 1800, consumption: 6200, level: '中端', buildYear: 2015 },
  { id: 3, name: '碧桂园翡翠湾', city: '广州', households: 3200, consumption: 5500, level: '中端', buildYear: 2020 }
])

const handleSearch = () => { loading.value = true; setTimeout(() => loading.value = false, 500) }
const viewDetail = (item) => { currentCommunity.value = item; showDetailModal.value = true }
</script>
