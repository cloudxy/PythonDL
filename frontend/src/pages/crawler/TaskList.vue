<template>
  <div class="space-y-6">
    <PageHeader title="爬虫采集" subtitle="管理数据采集任务">
      <template #actions>
        <Button variant="primary" @click="showAddModal = true">新建任务</Button>
      </template>
    </PageHeader>
    <Breadcrumb :items="[{ label: '爬虫采集' }]" />

    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
      <StatCard label="运行中" :value="stats.running" color="primary" />
      <StatCard label="已完成" :value="stats.completed" color="success" />
      <StatCard label="失败" :value="stats.failed" color="danger" />
      <StatCard label="待执行" :value="stats.pending" color="warning" />
    </div>

    <Card>
      <Table :columns="columns" :data="tasks" :loading="loading">
        <template #cell-status="{ row }">
          <Badge :variant="getStatusVariant(row.status)">{{ getStatusText(row.status) }}</Badge>
        </template>
        <template #cell-progress="{ row }">
          <Progress :value="row.progress" :color="row.status === 'running' ? 'primary' : 'success'" />
        </template>
        <template #cell-lastRun="{ row }">
          {{ formatDate(row.lastRun) }}
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button v-if="row.status === 'running'" variant="warning" size="sm" @click="stopTask(row)">停止</Button>
            <Button v-else variant="success" size="sm" @click="startTask(row)">启动</Button>
            <Button variant="ghost" size="sm" @click="viewLogs(row)">日志</Button>
            <Button variant="ghost" size="sm" @click="editTask(row)">编辑</Button>
            <Button variant="danger" size="sm" @click="deleteTask(row)">删除</Button>
          </div>
        </template>
      </Table>
    </Card>

    <Modal v-model="showAddModal" :title="editingTask ? '编辑任务' : '新建任务'" size="lg" :show-default-footer="true" @confirm="handleSubmit">
      <FormSection title="基本信息">
        <Input v-model="taskForm.name" label="任务名称" required />
        <Select v-model="taskForm.type" label="任务类型" :options="typeOptions" />
        <Input v-model="taskForm.url" label="目标URL" />
      </FormSection>
      <FormSection title="执行配置">
        <Select v-model="taskForm.schedule" label="执行周期" :options="scheduleOptions" />
        <Input v-model="taskForm.cron" label="Cron表达式" hint="如：0 0 * * * (每天零点)" />
        <Toggle v-model="taskForm.enabled" label="启用任务" />
      </FormSection>
    </Modal>

    <Modal v-model="showLogsModal" title="任务日志" size="xl">
      <div v-if="currentLogs.length" class="bg-secondary-900 text-green-400 p-4 rounded-lg font-mono text-sm h-96 overflow-auto">
        <div v-for="(log, index) in currentLogs" :key="index" class="mb-1">{{ log }}</div>
      </div>
      <div v-else class="h-64 flex items-center justify-center text-secondary-400">暂无日志</div>
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
import FormSection from '@/components/ui/FormSection.vue'
import Progress from '@/components/ui/Progress.vue'
import Toggle from '@/components/ui/Toggle.vue'
import StatCard from '@/components/ui/StatCard.vue'

const loading = ref(false)
const showAddModal = ref(false)
const showLogsModal = ref(false)
const editingTask = ref(null)
const currentLogs = ref([])

const stats = reactive({ running: 2, completed: 45, failed: 3, pending: 8 })

const columns = [
  { key: 'id', label: 'ID' },
  { key: 'name', label: '任务名称' },
  { key: 'type', label: '类型' },
  { key: 'status', label: '状态' },
  { key: 'progress', label: '进度' },
  { key: 'schedule', label: '执行周期' },
  { key: 'lastRun', label: '最后执行' }
]

const tasks = ref([
  { id: 1, name: '股票数据采集', type: '金融', status: 'running', progress: 65, schedule: '每日', lastRun: '2024-01-15 10:00:00' },
  { id: 2, name: '气象数据同步', type: '气象', status: 'completed', progress: 100, schedule: '每小时', lastRun: '2024-01-15 09:00:00' },
  { id: 3, name: '新闻资讯采集', type: '资讯', status: 'pending', progress: 0, schedule: '每6小时', lastRun: '2024-01-14 18:00:00' },
  { id: 4, name: '电商价格监控', type: '电商', status: 'failed', progress: 30, schedule: '每日', lastRun: '2024-01-15 08:00:00' }
])

const taskForm = reactive({ name: '', type: '', url: '', schedule: '', cron: '', enabled: true })
const typeOptions = [{ value: 'finance', label: '金融' }, { value: 'weather', label: '气象' }, { value: 'news', label: '资讯' }, { value: 'ecommerce', label: '电商' }]
const scheduleOptions = [{ value: 'hourly', label: '每小时' }, { value: 'daily', label: '每日' }, { value: 'weekly', label: '每周' }, { value: 'custom', label: '自定义' }]

const getStatusVariant = (status) => ({ running: 'primary', completed: 'success', failed: 'danger', pending: 'warning' }[status] || 'secondary')
const getStatusText = (status) => ({ running: '运行中', completed: '已完成', failed: '失败', pending: '待执行' }[status] || status)
const formatDate = (date) => date ? new Date(date).toLocaleString('zh-CN') : '-'

const startTask = (task) => { task.status = 'running'; task.progress = 0 }
const stopTask = (task) => { task.status = 'pending' }
const viewLogs = (task) => { currentLogs.value = [`[${new Date().toISOString()}] 开始执行任务: ${task.name}`, `[${new Date().toISOString()}] 正在获取数据...`, `[${new Date().toISOString()}] 数据获取完成`]; showLogsModal.value = true }
const editTask = (task) => { editingTask.value = task; Object.assign(taskForm, task); showAddModal.value = true }
const deleteTask = (task) => { tasks.value = tasks.value.filter(t => t.id !== task.id) }
const handleSubmit = () => { showAddModal.value = false }
</script>
