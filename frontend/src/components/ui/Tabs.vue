<template>
  <div class="border-b border-secondary-200">
    <nav class="flex gap-4">
      <button
        v-for="tab in tabs"
        :key="tab.value"
        @click="handleSelect(tab.value)"
        :class="[
          'px-4 py-3 text-sm font-medium border-b-2 transition-colors',
          modelValue === tab.value
            ? 'border-primary-600 text-primary-600'
            : 'border-transparent text-secondary-500 hover:text-secondary-700 hover:border-secondary-300'
        ]"
      >
        {{ tab.label }}
      </button>
    </nav>
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
