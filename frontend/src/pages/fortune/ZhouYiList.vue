<template>
  <div class="space-y-6">
    <PageHeader title="周易数据" subtitle="周易六十四卦数据管理" />

    <Breadcrumb :items="[{ label: '看相算命' }, { label: '周易数据' }]" />

    <Card>
      <Table :columns="columns" :data="zhouyiData" :loading="loading">
        <template #cell-element="{ row }">
          <Badge :variant="getElementVariant(row.element)">{{ row.element }}</Badge>
        </template>
        <template #cell-nature="{ row }">
          <Badge :variant="row.nature === '吉' ? 'success' : 'danger'">{{ row.nature }}</Badge>
        </template>
        <template #actions="{ row }">
          <Button variant="ghost" size="sm" @click="viewDetail(row)">详情</Button>
        </template>
      </Table>
    </Card>

    <Modal v-model="showDetailModal" title="卦象详情" size="lg">
      <div v-if="currentHexagram" class="space-y-4">
        <div class="text-center">
          <h3 class="text-2xl font-bold">{{ currentHexagram.name }}</h3>
          <p class="text-secondary-500">{{ currentHexagram.symbol }}</p>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <div><span class="text-secondary-500">上卦：</span>{{ currentHexagram.upperTrigram }}</div>
          <div><span class="text-secondary-500">下卦：</span>{{ currentHexagram.lowerTrigram }}</div>
        </div>
        <div>
          <h4 class="font-semibold mb-2">卦辞</h4>
          <p class="text-secondary-600">{{ currentHexagram.text }}</p>
        </div>
        <div>
          <h4 class="font-semibold mb-2">象辞</h4>
          <p class="text-secondary-600">{{ currentHexagram.image }}</p>
        </div>
        <div>
          <h4 class="font-semibold mb-2">解释</h4>
          <p class="text-secondary-600">{{ currentHexagram.interpretation }}</p>
        </div>
      </div>
    </Modal>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'

const loading = ref(false)
const showDetailModal = ref(false)
const currentHexagram = ref(null)

const columns = [
  { key: 'id', label: '序号' },
  { key: 'name', label: '卦名' },
  { key: 'symbol', label: '卦象' },
  { key: 'element', label: '五行' },
  { key: 'nature', label: '性质' },
  { key: 'upperTrigram', label: '上卦' },
  { key: 'lowerTrigram', label: '下卦' }
]

const zhouyiData = ref([
  { id: 1, name: '乾', symbol: '☰', element: '金', nature: '吉', upperTrigram: '乾', lowerTrigram: '乾', text: '元亨利贞。', image: '天行健，君子以自强不息。', interpretation: '乾卦象征天，代表刚健、创始、领导。' },
  { id: 2, name: '坤', symbol: '☷', element: '土', nature: '吉', upperTrigram: '坤', lowerTrigram: '坤', text: '元亨，利牝马之贞。', image: '地势坤，君子以厚德载物。', interpretation: '坤卦象征地，代表柔顺、承载、包容。' },
  { id: 3, name: '屯', symbol: '☳☵', element: '水', nature: '凶', upperTrigram: '坎', lowerTrigram: '震', text: '元亨利贞，勿用有攸往，利建侯。', image: '云雷屯，君子以经纶。', interpretation: '屯卦象征初生，代表困难与希望并存。' }
])

const getElementVariant = (element) => {
  const variants = { 金: 'warning', 木: 'success', 水: 'primary', 火: 'danger', 土: 'secondary' }
  return variants[element] || 'secondary'
}

const viewDetail = (item) => { currentHexagram.value = item; showDetailModal.value = true }
</script>
