<template>
  <div class="relative inline-block" ref="dropdownRef">
    <div @click="toggleDropdown">
      <slot name="trigger">
        <Button variant="secondary">
          {{ label }}
          <svg class="w-4 h-4 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
          </svg>
        </Button>
      </slot>
    </div>
    <Transition
      enter-active-class="transition ease-out duration-100"
      enter-from-class="transform opacity-0 scale-95"
      enter-to-class="transform opacity-100 scale-100"
      leave-active-class="transition ease-in duration-75"
      leave-from-class="transform opacity-100 scale-100"
      leave-to-class="transform opacity-0 scale-95"
    >
      <div
        v-show="isOpen"
        :class="[
          'absolute z-50 mt-2 bg-white rounded-lg shadow-lg border border-secondary-100 py-1',
          positionClass,
          sizeClass
        ]"
      >
        <slot>
          <a
            v-for="(item, index) in items"
            :key="index"
            href="#"
            @click.prevent="handleSelect(item)"
            :class="[
              'block px-4 py-2 text-sm text-secondary-700 hover:bg-secondary-50 transition-colors',
              item.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
            ]"
          >
            <div class="flex items-center gap-2">
              <component v-if="item.icon" :is="item.icon" class="w-4 h-4" />
              {{ item.label }}
            </div>
          </a>
        </slot>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import Button from './Button.vue'

const props = defineProps({
  label: {
    type: String,
    default: '选项'
  },
  items: {
    type: Array,
    default: () => []
  },
  position: {
    type: String,
    default: 'bottom-start',
    validator: (v) => ['bottom-start', 'bottom-end', 'top-start', 'top-end'].includes(v)
  },
  size: {
    type: String,
    default: 'md',
    validator: (v) => ['sm', 'md', 'lg'].includes(v)
  }
})

const emit = defineEmits(['select'])

const isOpen = ref(false)
const dropdownRef = ref(null)

const positionClass = computed(() => {
  const classes = {
    'bottom-start': 'left-0',
    'bottom-end': 'right-0',
    'top-start': 'left-0 bottom-full',
    'top-end': 'right-0 bottom-full'
  }
  return classes[props.position]
})

const sizeClass = computed(() => {
  const classes = {
    sm: 'w-32',
    md: 'w-48',
    lg: 'w-64'
  }
  return classes[props.size]
})

const toggleDropdown = () => {
  isOpen.value = !isOpen.value
}

const handleSelect = (item) => {
  if (!item.disabled) {
    emit('select', item)
    isOpen.value = false
  }
}

const handleClickOutside = (event) => {
  if (dropdownRef.value && !dropdownRef.value.contains(event.target)) {
    isOpen.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>
