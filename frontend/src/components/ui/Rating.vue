<template>
  <div class="flex items-center gap-1">
    <svg
      v-for="i in maxStars"
      :key="i"
      :class="[
        'w-5 h-5 cursor-pointer transition-colors',
        i <= modelValue ? 'text-warning-400' : 'text-secondary-300'
      ]"
      fill="currentColor"
      viewBox="0 0 20 20"
      @click="handleClick(i)"
      @mouseenter="hoverValue = i"
      @mouseleave="hoverValue = 0"
    >
      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
    </svg>
    <span v-if="showValue" class="text-sm text-secondary-600 ml-2">
      {{ hoverValue || modelValue }} / {{ maxStars }}
    </span>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  modelValue: {
    type: Number,
    default: 0
  },
  maxStars: {
    type: Number,
    default: 5
  },
  readonly: {
    type: Boolean,
    default: false
  },
  showValue: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:modelValue'])

const hoverValue = ref(0)

const handleClick = (value) => {
  if (!props.readonly) {
    emit('update:modelValue', value)
  }
}
</script>
