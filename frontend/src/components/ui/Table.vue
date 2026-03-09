<template>
  <div class="overflow-hidden">
    <div class="overflow-x-auto">
      <table class="data-table">
        <thead>
          <tr>
            <th v-if="selectable" class="w-12">
              <input
                type="checkbox"
                :checked="isAllSelected"
                :indeterminate="isIndeterminate"
                @change="toggleSelectAll"
                class="w-4 h-4 rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
              />
            </th>
            <th
              v-for="column in columns"
              :key="column.key"
              :class="[
                column.headerClass,
                column.sortable ? 'cursor-pointer select-none hover:bg-secondary-100' : ''
              ]"
              @click="column.sortable && handleSort(column.key)"
            >
              <div class="flex items-center gap-2">
                <span>{{ column.label }}</span>
                <span v-if="column.sortable && sortKey === column.key" class="text-primary-600">
                  <svg v-if="sortOrder === 'asc'" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7" />
                  </svg>
                  <svg v-else class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                  </svg>
                </span>
              </div>
            </th>
            <th v-if="$slots.actions" class="text-right">操作</th>
          </tr>
        </thead>
        <tbody>
          <template v-if="loading">
            <tr v-for="i in 5" :key="i">
              <td v-if="selectable" class="w-12">
                <div class="w-4 h-4 bg-secondary-200 rounded animate-pulse"></div>
              </td>
              <td v-for="column in columns" :key="column.key">
                <div class="h-4 bg-secondary-200 rounded animate-pulse"></div>
              </td>
              <td v-if="$slots.actions">
                <div class="flex justify-end gap-2">
                  <div class="w-16 h-8 bg-secondary-200 rounded animate-pulse"></div>
                </div>
              </td>
            </tr>
          </template>
          <template v-else-if="data.length === 0">
            <tr>
              <td :colspan="columnCount" class="text-center py-12">
                <div class="flex flex-col items-center">
                  <svg class="w-12 h-12 text-secondary-300 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                  </svg>
                  <p class="text-secondary-500">{{ emptyText }}</p>
                </div>
              </td>
            </tr>
          </template>
          <template v-else>
            <tr
              v-for="(row, index) in data"
              :key="row[rowKey] || index"
              :class="rowClass(row, index)"
            >
              <td v-if="selectable" class="w-12">
                <input
                  type="checkbox"
                  :checked="isSelected(row)"
                  @change="toggleSelect(row)"
                  class="w-4 h-4 rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
                />
              </td>
              <td
                v-for="column in columns"
                :key="column.key"
                :class="column.cellClass"
              >
                <slot :name="`cell-${column.key}`" :row="row" :value="row[column.key]">
                  {{ formatValue(row[column.key], column) }}
                </slot>
              </td>
              <td v-if="$slots.actions" class="text-right">
                <slot name="actions" :row="row" :index="index" />
              </td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>

    <!-- Pagination -->
    <div v-if="showPagination" class="px-6 py-4 border-t border-secondary-100 flex items-center justify-between">
      <div class="text-sm text-secondary-500">
        显示 {{ startIndex + 1 }} - {{ endIndex }} 条，共 {{ total }} 条
      </div>
      <div class="flex items-center gap-2">
        <select
          v-model="localPageSize"
          class="form-input py-1.5 text-sm w-20"
          @change="handlePageSizeChange"
        >
          <option v-for="size in pageSizeOptions" :key="size" :value="size">{{ size }}</option>
        </select>
        <nav class="flex items-center gap-1">
          <button
            :disabled="currentPage === 1"
            @click="handlePageChange(1)"
            class="p-2 rounded-lg text-secondary-500 hover:bg-secondary-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
            </svg>
          </button>
          <button
            :disabled="currentPage === 1"
            @click="handlePageChange(currentPage - 1)"
            class="p-2 rounded-lg text-secondary-500 hover:bg-secondary-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <template v-for="page in visiblePages" :key="page">
            <span v-if="page === '...'" class="px-2 text-secondary-400">...</span>
            <button
              v-else
              @click="handlePageChange(page)"
              :class="[
                'w-8 h-8 rounded-lg text-sm font-medium transition-colors',
                page === currentPage
                  ? 'bg-primary-600 text-white'
                  : 'text-secondary-600 hover:bg-secondary-100'
              ]"
            >
              {{ page }}
            </button>
          </template>
          <button
            :disabled="currentPage === totalPages"
            @click="handlePageChange(currentPage + 1)"
            class="p-2 rounded-lg text-secondary-500 hover:bg-secondary-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
            </svg>
          </button>
          <button
            :disabled="currentPage === totalPages"
            @click="handlePageChange(totalPages)"
            class="p-2 rounded-lg text-secondary-500 hover:bg-secondary-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
            </svg>
          </button>
        </nav>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({
  columns: {
    type: Array,
    required: true
  },
  data: {
    type: Array,
    default: () => []
  },
  rowKey: {
    type: String,
    default: 'id'
  },
  loading: {
    type: Boolean,
    default: false
  },
  emptyText: {
    type: String,
    default: '暂无数据'
  },
  selectable: {
    type: Boolean,
    default: false
  },
  selectedKeys: {
    type: Array,
    default: () => []
  },
  rowClass: {
    type: Function,
    default: () => ''
  },
  // Pagination
  showPagination: {
    type: Boolean,
    default: true
  },
  total: {
    type: Number,
    default: 0
  },
  page: {
    type: Number,
    default: 1
  },
  pageSize: {
    type: Number,
    default: 10
  },
  pageSizeOptions: {
    type: Array,
    default: () => [10, 20, 50, 100]
  }
})

const emit = defineEmits(['update:selectedKeys', 'update:page', 'update:pageSize', 'sort', 'page-change'])

const sortKey = ref('')
const sortOrder = ref('asc')
const localPageSize = ref(props.pageSize)

const columnCount = computed(() => {
  let count = props.columns.length
  if (props.selectable) count++
  if (props.$slots?.actions) count++
  return count
})

const currentPage = computed(() => props.page)
const totalPages = computed(() => Math.ceil(props.total / localPageSize.value))
const startIndex = computed(() => (currentPage.value - 1) * localPageSize.value)
const endIndex = computed(() => Math.min(startIndex.value + localPageSize.value, props.total))

const visiblePages = computed(() => {
  const pages = []
  const total = totalPages.value
  const current = currentPage.value

  if (total <= 7) {
    for (let i = 1; i <= total; i++) {
      pages.push(i)
    }
  } else {
    if (current <= 4) {
      for (let i = 1; i <= 5; i++) {
        pages.push(i)
      }
      pages.push('...', total)
    } else if (current >= total - 3) {
      pages.push(1, '...')
      for (let i = total - 4; i <= total; i++) {
        pages.push(i)
      }
    } else {
      pages.push(1, '...')
      for (let i = current - 1; i <= current + 1; i++) {
        pages.push(i)
      }
      pages.push('...', total)
    }
  }

  return pages
})

const isAllSelected = computed(() => {
  return props.data.length > 0 && props.selectedKeys.length === props.data.length
})

const isIndeterminate = computed(() => {
  return props.selectedKeys.length > 0 && props.selectedKeys.length < props.data.length
})

const isSelected = (row) => {
  return props.selectedKeys.includes(row[props.rowKey])
}

const toggleSelect = (row) => {
  const key = row[props.rowKey]
  const keys = [...props.selectedKeys]
  const index = keys.indexOf(key)
  if (index > -1) {
    keys.splice(index, 1)
  } else {
    keys.push(key)
  }
  emit('update:selectedKeys', keys)
}

const toggleSelectAll = () => {
  if (isAllSelected.value) {
    emit('update:selectedKeys', [])
  } else {
    emit('update:selectedKeys', props.data.map(row => row[props.rowKey]))
  }
}

const handleSort = (key) => {
  if (sortKey.value === key) {
    sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortKey.value = key
    sortOrder.value = 'asc'
  }
  emit('sort', { key: sortKey.value, order: sortOrder.value })
}

const handlePageChange = (page) => {
  emit('update:page', page)
  emit('page-change', { page, pageSize: localPageSize.value })
}

const handlePageSizeChange = () => {
  emit('update:pageSize', localPageSize.value)
  emit('page-change', { page: 1, pageSize: localPageSize.value })
}

const formatValue = (value, column) => {
  if (column.formatter) {
    return column.formatter(value)
  }
  if (value === null || value === undefined) {
    return '-'
  }
  return value
}

watch(() => props.pageSize, (val) => {
  localPageSize.value = val
})
</script>
