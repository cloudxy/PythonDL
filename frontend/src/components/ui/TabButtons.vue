<template>
  <div class="flex items-center gap-2">
    <button
      v-for="tab in tabs"
      :key="tab.value"
      @click="handleSelect(tab.value)"
      :class="[
        'px-4 py-2 text-sm font-medium rounded-lg transition-colors',
        modelValue === tab.value
          ? 'bg-primary-100 text-primary-700'
          : 'text-secondary-600 hover:bg-secondary-100'
      ]"
    >
      {{ tab.label }}
    </button>
  </div>
</template>

<script setup>
const props = defineProps({
  modelValue: {
    type: [String, Number],
    default: ''
  },
  tabs: {
    type: Array,
    required: true
  }
})

const emit = defineEmits(['update:modelValue'])

const handleSelect = (value) => {
  emit('update:modelValue', value)
}
</script>
