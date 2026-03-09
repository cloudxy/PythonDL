<template>
  <div class="space-y-6">
    <PageHeader title="经济指标" subtitle="经济指标数据查看" />
    <Breadcrumb :items="[{ label: '消费分析' }, { label: '经济指标' }]" />
    
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <StatCard label="GDP增速" value="5.2" suffix="%" color="primary" :change="0.3" />
      <StatCard label="CPI涨幅" value="0.2" suffix="%" color="success" :change="-0.5" />
      <StatCard label="PMI指数" value="50.3" color="warning" :change="1.2" />
      <StatCard label="失业率" value="5.1" suffix="%" color="danger" :change="-0.2" />
    </div>

    <Card title="经济指标列表">
      <Table :columns="columns" :data="indicators" :loading="loading">
        <template #cell-trend="{ row }">
          <span :class="row.trend >= 0 ? 'text-success-600' : 'text-danger-600'">{{ row.trend >= 0 ? '↑' : '↓' }} {{ Math.abs(row.trend) }}</span>
        </template>
        <template #cell-status="{ row }">
          <Badge :variant="row.status === 'good' ? 'success' : row.status === 'warning' ? 'warning' : 'danger'">{{ row.status === 'good' ? '良好' : row.status === 'warning' ? '一般' : '较差' }}</Badge>
        </template>
      </Table>
    </Card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import StatCard from '@/components/ui/StatCard.vue'

const loading = ref(false)

const columns = [
  { key: 'id', label: 'ID' },
  { key: 'name', label: '指标名称' },
  { key: 'value', label: '当前值' },
  { key: 'unit', label: '单位' },
  { key: 'trend', label: '变化' },
  { key: 'status', label: '状态' },
  { key: 'updateTime', label: '更新时间' }
]

const indicators = ref([
  { id: 1, name: 'GDP增速', value: 5.2, unit: '%', trend: 0.3, status: 'good', updateTime: '2024-01-15' },
  { id: 2, name: 'CPI涨幅', value: 0.2, unit: '%', trend: -0.5, status: 'good', updateTime: '2024-01-15' },
  { id: 3, name: 'PPI涨幅', value: -2.7, unit: '%', trend: -0.3, status: 'warning', updateTime: '2024-01-15' },
  { id: 4, name: 'PMI指数', value: 50.3, unit: '', trend: 1.2, status: 'good', updateTime: '2024-01-15' },
  { id: 5, name: '社会消费品零售总额增速', value: 7.2, unit: '%', trend: 2.1, status: 'good', updateTime: '2024-01-15' }
])
</script>
