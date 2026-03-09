<template>
  <div class="space-y-6">
    <PageHeader title="面相数据管理" subtitle="管理面相分析数据">
      <template #actions>
        <Button variant="primary" @click="showAddModal = true">新增数据</Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[{ label: '看相算命' }, { label: '面相数据' }]" />

    <Card>
      <Table :columns="columns" :data="faceData" :loading="loading">
        <template #cell-part="{ row }">
          <Badge variant="primary">{{ row.part }}</Badge>
        </template>
        <template #cell-importance="{ row }">
          <Progress :value="row.importance" color="primary" />
        </template>
        <template #actions="{ row }">
          <Button variant="ghost" size="sm" @click="editItem(row)">编辑</Button>
          <Button variant="danger" size="sm" @click="deleteItem(row)">删除</Button>
        </template>
      </Table>
    </Card>

    <Modal v-model="showAddModal" :title="editingItem ? '编辑数据' : '新增数据'" size="lg" :show-default-footer="true" @confirm="handleSubmit">
      <FormSection>
        <Input v-model="form.part" label="部位名称" required />
        <Input v-model="form.location" label="位置描述" />
        <Slider v-model="form.importance" label="重要程度" :max="100" />
        <Textarea v-model="form.meaning" label="面相含义" />
        <Textarea v-model="form.goodSigns" label="吉相特征" />
        <Textarea v-model="form.badSigns" label="凶相特征" />
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
import Textarea from '@/components/ui/Textarea.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import Progress from '@/components/ui/Progress.vue'
import Slider from '@/components/ui/Slider.vue'

const loading = ref(false)
const showAddModal = ref(false)
const editingItem = ref(null)

const columns = [
  { key: 'id', label: 'ID' },
  { key: 'part', label: '部位' },
  { key: 'location', label: '位置' },
  { key: 'importance', label: '重要程度' },
  { key: 'meaning', label: '含义' }
]

const faceData = ref([
  { id: 1, part: '额头', location: '眉上发下', importance: 85, meaning: '代表早年运势与智慧', goodSigns: '宽阔饱满', badSigns: '低窄凹陷' },
  { id: 2, part: '眉毛', location: '眼上', importance: 70, meaning: '代表兄弟姐妹情谊', goodSigns: '清秀修长', badSigns: '杂乱短促' },
  { id: 3, part: '眼睛', location: '面部中央', importance: 95, meaning: '代表心性与精神', goodSigns: '明亮有神', badSigns: '浑浊无光' }
])

const form = reactive({ part: '', location: '', importance: 50, meaning: '', goodSigns: '', badSigns: '' })

const editItem = (item) => { editingItem.value = item; Object.assign(form, item); showAddModal.value = true }
const deleteItem = (item) => { faceData.value = faceData.value.filter(d => d.id !== item.id) }
const handleSubmit = () => { showAddModal.value = false }
</script>
