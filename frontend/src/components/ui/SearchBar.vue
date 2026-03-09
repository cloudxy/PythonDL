<template>
  <div class="flex items-center gap-4 mb-6">
    <div class="flex-1 relative">
      <svg class="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-secondary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
      <input
        v-model="searchValue"
        type="text"
        :placeholder="placeholder"
        class="form-input pl-10"
        @keyup.enter="handleSearch"
      />
    </div>
    <Button v-if="showSearchButton" @click="handleSearch">
      搜索
    </Button>
    <Button v-if="showAddButton" variant="primary" @click="$emit('add')">
      <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
      </svg>
      新增
    </Button>
    <slot />
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import Button from './Button.vue'

const props = defineProps({
  modelValue: {
    type: String,
    default: ''
  },
  placeholder: {
    type: String,
    default: '搜索...'
  },
  showSearchButton: {
    type: Boolean,
    default: true
  },
  showAddButton: {
    type: Boolean,
    default: true
  }
})

const emit = defineEmits(['update:modelValue', 'search', 'add'])

const searchValue = ref(props.modelValue)

watch(() => props.modelValue, (val) => {
  searchValue.value = val
})

const handleSearch = () => {
  emit('update:modelValue', searchValue.value)
  emit('search', searchValue.value)
}
</script>
